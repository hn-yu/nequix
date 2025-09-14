import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from nequix.layer_norm import RMSLayerNorm

# Covalent radii in nanometers. Values are converted to Å below.
# Data originally from https://doi.org/10.1063/1.1725697.
_COVALENT_RADII_NM = {
    1: 0.025,
    2: 0.028,
    3: 0.145,
    4: 0.105,
    5: 0.085,
    6: 0.07,
    7: 0.065,
    8: 0.06,
    9: 0.05,
    10: 0.058,
    11: 0.18,
    12: 0.15,
    13: 0.125,
    14: 0.11,
    15: 0.1,
    16: 0.1,
    17: 0.1,
    18: 0.106,
    19: 0.22,
    20: 0.18,
    21: 0.16,
    22: 0.14,
    23: 0.135,
    24: 0.14,
    25: 0.14,
    26: 0.14,
    27: 0.135,
    28: 0.135,
    29: 0.135,
    30: 0.135,
    31: 0.13,
    32: 0.125,
    33: 0.115,
    34: 0.115,
    35: 0.115,
    36: 0.116,
    37: 0.235,
    38: 0.2,
    39: 0.18,
    40: 0.155,
    41: 0.145,
    42: 0.145,
    43: 0.135,
    44: 0.13,
    45: 0.135,
    46: 0.14,
    47: 0.16,
    48: 0.155,
    49: 0.155,
    50: 0.145,
    51: 0.145,
    52: 0.14,
    53: 0.14,
    54: 0.14,
    55: 0.244,
    56: 0.215,
    57: 0.207,
    58: 0.204,
    59: 0.203,
    60: 0.201,
    61: 0.199,
    62: 0.198,
    63: 0.198,
    64: 0.196,
    65: 0.194,
    66: 0.192,
    67: 0.192,
    68: 0.189,
    69: 0.19,
    70: 0.187,
    71: 0.187,
    72: 0.175,
    73: 0.17,
    74: 0.162,
    75: 0.151,
    76: 0.144,
    77: 0.141,
    78: 0.136,
    79: 0.136,
    80: 0.132,
    81: 0.145,
    82: 0.146,
    83: 0.148,
    84: 0.14,
    85: 0.15,
    86: 0.15,
    87: 0.26,
    88: 0.221,
    89: 0.215,
    90: 0.206,
    91: 0.2,
    92: 0.196,
    93: 0.19,
    94: 0.187,
}
COVALENT_RADII = {k: v * 10.0 for k, v in _COVALENT_RADII_NM.items()}


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


def zbl_pair_energy(r: jax.Array, z1: jax.Array, z2: jax.Array, cutoff: jax.Array) -> jax.Array:
    """Compute the ZBL repulsive potential for a pair of atoms.

    Parameters
    ----------
    r : jax.Array
        Distance between atoms in Å.
    z1, z2 : jax.Array
        Atomic numbers of the two atoms.
    cutoff : jax.Array
        Cutoff radius (sum of covalent radii) in Å.

    Returns
    -------
    jax.Array
        Repulsive energy in eV.
    """
    # Prevent division by zero for coincident atoms
    r = jnp.where(r > 0.0, r, 1e-12)

    a0 = 0.529177210903  # Bohr radius in Å
    a = 0.8854 * a0 / (jnp.power(z1, 0.23) + jnp.power(z2, 0.23))
    d = r / a
    screening = (
        0.1818 * jnp.exp(-3.2 * d)
        + 0.5099 * jnp.exp(-0.9423 * d)
        + 0.2802 * jnp.exp(-0.4029 * d)
        + 0.02817 * jnp.exp(-0.2016 * d)
    )
    ke = 14.399652  # 1/(4*pi*eps0) in eV*Å
    energy = ke * z1 * z2 / r * screening

    phi = jnp.where(r < cutoff, 0.5 * (jnp.cos(jnp.pi * r / cutoff) + 1.0), 0.0)
    return energy * phi


class Linear(eqx.Module):
    weights: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = jax.random.normal(key, (in_size, out_size)) * scale
        self.bias = jnp.zeros(out_size) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(eqx.Module):
    layers: list[Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activation=jax.nn.silu,
        *,
        init_scale: float = 1.0,
        use_bias: bool = False,
        key: jax.Array,
    ):
        self.activation = activation

        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NequixConvolution(eqx.Module):
    output_irreps: e3nn.Irreps = eqx.field(static=True)
    index_weights: bool = eqx.field(static=True)
    avg_n_neighbors: float = eqx.field(static=True)

    radial_mlp: MLP
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear
    skip: e3nn.equinox.Linear
    layer_norm: Optional[RMSLayerNorm]

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = True,
        layer_norm: bool = False,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights

        tp_irreps = e3nn.tensor_product(input_irreps, sh_irreps, filter_ir_out=output_irreps)

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            key=k1,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=output_irreps,
            key=k3,
        )

        # skip connection has per-species weights
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True,
            key=k4,
        )

        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                irreps=output_irreps,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        messages = self.linear_1(features)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(jax.lax.stop_gradient(self.avg_n_neighbors))

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )


class Nequix(eqx.Module):
    lmax: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    atomic_numbers: jax.Array = eqx.field(static=True)
    covalent_radii: jax.Array = eqx.field(static=True)

    atom_energies: jax.Array
    layers: list[NequixConvolution]
    readout: e3nn.equinox.Linear

    def __init__(
        self,
        key,
        atomic_numbers: Sequence[int],
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = True,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
    ):
        n_species = len(atomic_numbers)
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atomic_numbers = jnp.array(atomic_numbers, dtype=jnp.float32)
        missing = [n for n in atomic_numbers if n not in COVALENT_RADII]
        if missing:
            raise ValueError(f"Missing covalent radius for atomic number(s): {missing}")
        self.covalent_radii = jnp.array(
            [COVALENT_RADII[n] for n in atomic_numbers], dtype=jnp.float32
        )
        self.atom_energies = (
            jnp.array(atom_energies)
            if atom_energies is not None
            else jnp.zeros(n_species, dtype=jnp.float32)
        )
        input_irreps = e3nn.Irreps(f"{n_species}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.layers = []

        key, *subkeys = jax.random.split(key, n_layers + 1)
        for i in range(n_layers):
            self.layers.append(
                NequixConvolution(
                    key=subkeys[i],
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps if i < n_layers - 1 else hidden_irreps.filter("0e"),
                    sh_irreps=sh_irreps,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_n_neighbors=avg_n_neighbors,
                    index_weights=index_weights,
                    layer_norm=layer_norm,
                )
            )

        self.readout = e3nn.equinox.Linear(
            irreps_in=hidden_irreps.filter("0e"), irreps_out="0e", key=key
        )

    def node_energies(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ):
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"), jax.nn.one_hot(species, self.n_species)
        )

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(displacements**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * polynomial_cutoff(
                r_norm,
                self.cutoff,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(
                features,
                species,
                sh,
                radial_basis,
                senders,
                receivers,
            )

        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * jax.lax.stop_gradient(self.scale) + jax.lax.stop_gradient(
            self.shift
        )

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(self.atom_energies[species, None])

        return node_energies.array

    def __call__(self, data: jraph.GraphsTuple):
        # compute forces and stress as gradient of total energy with respect to positions and strain
        def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
            positions, eps = positions_eps
            eps_sym = (eps + eps.swapaxes(1, 2)) / 2
            eps_sym_per_node = jnp.repeat(
                eps_sym,
                data.n_node,
                axis=0,
                total_repeat_length=data.nodes["positions"].shape[0],
            )
            # apply strain to positions and cell
            positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
            cell = data.globals["cell"] + jnp.einsum("bij,bjk->bik", data.globals["cell"], eps_sym)
            cell_per_edge = jnp.repeat(
                cell,
                data.n_edge,
                axis=0,
                total_repeat_length=data.edges["shifts"].shape[0],
            )
            offsets = jnp.einsum("ij,ijk->ik", data.edges["shifts"], cell_per_edge)
            r = positions[data.senders] - positions[data.receivers] + offsets
            node_energies = self.node_energies(
                r, data.nodes["species"], data.senders, data.receivers
            )

            z1 = self.atomic_numbers[data.nodes["species"][data.senders]]
            z2 = self.atomic_numbers[data.nodes["species"][data.receivers]]
            r1 = self.covalent_radii[data.nodes["species"][data.senders]]
            r2 = self.covalent_radii[data.nodes["species"][data.receivers]]
            r_norm = jnp.linalg.norm(r, axis=-1)
            edge_energy = zbl_pair_energy(r_norm, z1, z2, r1 + r2)
            edge_mask = jraph.get_edge_padding_mask(data)
            edge_energy = jnp.where(edge_mask, edge_energy, 0.0)

            num_nodes = node_energies.shape[0]
            node_zbl = (
                jraph.segment_sum(edge_energy, data.senders, num_segments=num_nodes)
                + jraph.segment_sum(edge_energy, data.receivers, num_segments=num_nodes)
            ) * 0.5
            node_energies = node_energies + node_zbl[:, None]

            return jnp.sum(node_energies), node_energies

        eps = jnp.zeros_like(data.globals["cell"])

        (minus_forces, virial), node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
            (data.nodes["positions"], eps)
        )

        # padded nodes may have nan forces, so we mask them
        node_mask = jraph.get_node_padding_mask(data)

        minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
        det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
        stress = virial / det

        # padded stress may be nan, so we mask them
        graph_mask = jraph.get_graph_padding_mask(data)
        stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

        return graph_energies[:, 0], -minus_forces, stress


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    # based on https://github.com/google-deepmind/jraph/blob/51f5990/jraph/_src/models.py#L209-L216
    n_graph = data.n_node.shape[0]
    # equivalent to jnp.sum(n_node), but jittable
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node)
    return node_gr_idx


def weight_decay_mask(model):
    """weight decay mask (only apply decay to linear weights)"""

    def is_layer(x):
        return isinstance(x, Linear) or isinstance(x, e3nn.equinox.Linear)

    def set_mask(x):
        if isinstance(x, Linear):
            mask = jax.tree.map(lambda _: True, x)
            mask = eqx.tree_at(lambda m: m.bias, mask, False)
            return mask
        elif isinstance(x, e3nn.equinox.Linear):
            return jax.tree.map(lambda _: True, x)
        else:
            return jax.tree.map(lambda _: False, x)

        return mask

    mask = jax.tree.map(set_mask, model, is_leaf=is_layer)
    return mask


def save_model(path: str, model: eqx.Module, config: dict):
    """Save a model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str) -> tuple[Nequix, dict]:
    """Load a model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = Nequix(
            key=jax.random.key(0),
            atomic_numbers=config["atomic_numbers"],
            hidden_irreps=config["hidden_irreps"],
            lmax=config["lmax"],
            cutoff=config["cutoff"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            index_weights=config["index_weights"],
            layer_norm=config["layer_norm"],
            shift=config["shift"],
            scale=config["scale"],
            avg_n_neighbors=config["avg_n_neighbors"],
            # NOTE: atom_energies will be in model weights
        )
        model = eqx.tree_deserialise_leaves(f, model)
        return model, config
