import numpy as np
import torch
from typing import Iterable, List, Tuple
from torch import Tensor
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform


class Style:
    """
    A collection of colors.
    """
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def set_seed(seed: int) -> None:
    """
    Set the random seed of NumPy and PyTorch.

    :param seed: The random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@functional_transform('MAP')
class MAP(BaseTransform):
    """
    Apply Maximal Axis Projection to the node features.

    :param norm_adj: Whether to normalize the adjacency matrix.
    :param k: The number of eigenvectors to use in the spectral embeddings.
    :param unique_sign: Whether to eliminate sign ambiguity.
    :param unique_basis: Whether to eliminate basis ambiguity.
    :param use_eig_val: Whether to incorporate eigenvalue information in the spectral embeddings.
    """
    def __init__(self, norm_adj: bool = True, k: int = 32, unique_sign: bool = True,
                 unique_basis: bool = True, use_eig_val: bool = True):
        self.normAdj = norm_adj
        self.k = k
        self.unique_sign = unique_sign
        self.unique_basis = unique_basis
        self.use_eig_val = use_eig_val

    def __call__(self, data):
        n = data.num_nodes = data.x.shape[0]
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:  # no edges in the graph
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        if self.normAdj:
            A = normalize_adjacency(A)
        E, U = torch.linalg.eigh(A)
        E = E.round(decimals=14)
        dim = min(n, self.k)
        _, mult = torch.unique(E[-dim:], return_counts=True)
        ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)]) + max(n - self.k, 0)
        if self.unique_sign:
            for i in range(mult.shape[0]):
                if mult[i] == 1:
                    U[:, ind[i]:ind[i + 1]] = unique_sign(U[:, ind[i]:ind[i + 1]])  # eliminate sign ambiguity
        if self.unique_basis:
            for i in range(mult.shape[0]):
                if mult[i] == 1:
                    continue  # single eigenvector, no basis ambiguity
                try:
                    U[:, ind[i]:ind[i + 1]] = unique_basis(U[:, ind[i]:ind[i + 1]])  # eliminate basis ambiguity
                except AssertionError:
                    continue  # assumption violated, skip
        if self.use_eig_val:
            Lambda = torch.nn.ReLU()(torch.diag(E))
            U = U @ torch.sqrt(Lambda)
        if n < self.k:
            zeros = torch.zeros([n, self.k - n])
            U = torch.cat([U, zeros], dim=-1)
        data.x = torch.cat([data.x, U[:, -self.k:]], dim=-1)  # last k eigenvectors
        return data


def normalize_adjacency(A: Tensor) -> Tensor:
    """
    Normalize the adjacency matrix of a graph.

    :param A: The adjacency matrix.
    """
    n = A.shape[0]
    assert list(A.shape) == [n, n]
    d = torch.sum(A, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    D_inv_sqrt[D_inv_sqrt == float("inf")] = 0.
    A = D_inv_sqrt @ A @ D_inv_sqrt
    A += torch.eye(n)
    return A


def count_multiplicities(dataset: Iterable) -> List[Tuple[int, int]]:
    """
    Count the number of eigenvalues with different multiplicities.

    :param dataset: The dataset.
    :return: List of (mult, count), where mult is the multiplicity and count is the number of eigenvalues with
        that multiplicity.
    """
    multiplicities = {}
    for data in dataset:
        n = data.num_nodes = data.x.shape[0]
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        A = normalize_adjacency(A)
        E, V = torch.linalg.eigh(A)
        _, eig_cnts = torch.unique(E.round(decimals=14), return_counts=True)
        for cnt in eig_cnts:
            if cnt.item() in multiplicities:
                multiplicities[cnt.item()] += 1
            else:
                multiplicities[cnt.item()] = 1
    return sorted(multiplicities.items(), key=lambda x: x[0])


def count_number_of_eigenvalues(dataset: Iterable) -> int:
    """
    Count the total number of eigenvalues in the dataset. Small graphs with no more than 5 nodes are ignored.

    :param dataset: The dataset.
    :return: The number of eigenvalues in the dataset.
    """
    count = 0
    for data in dataset:
        n = data.num_nodes = data.x.shape[0]
        if n <= 5:
            continue
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        A = normalize_adjacency(A)
        E, V = torch.linalg.eigh(A)
        eig_vals = torch.unique(E.round(decimals=14))
        count += len(eig_vals)
    return count


def unique_sign(U: Tensor) -> Tensor:
    """
    Eliminating sign ambiguity of the input eigenvectors.

    >>> U = Tensor([[1, -1, 4], [2, -2, 5], [3, -3, -6]])
    >>> unique_sign(U)
    tensor([[ 1.,  1.,  4.],
            [ 2.,  2.,  5.],
            [ 3.,  3., -6.]])
    >>> U = Tensor([[2, -2, 5], [3, -3, -6], [1, -1, 4]])
    >>> unique_sign(U)
    tensor([[ 2.,  2.,  5.],
            [ 3.,  3., -6.],
            [ 1.,  1.,  4.]])

    :param U: Tensor of shape [n, d]. Each column of U is an eigenvector.
    :return: Tensor of shape [n, d].
    """
    n, d = U.shape
    for i in range(d):
        u = U[:, i].view(n, 1)
        P = u @ u.T.view(1, n)
        E = torch.eye(n)
        J = torch.ones(n)
        Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
        Pe = [i for i in enumerate(Pe)]
        Pe.sort(key=lambda x: x[1])
        indices = [i[0] for i in Pe]
        lengths = [i[1] for i in Pe]
        _, counts = np.unique(lengths, return_counts=True)
        step = 0
        X = torch.zeros([len(counts), n])
        for j in range(len(counts)):
            for _ in range(counts[j]):
                X[j] += E[indices[step]]
                step += 1
            X[j] += 10 * J
        u_0, x = torch.zeros(n), torch.zeros(n)
        flag = True
        for j in range(len(counts)):
            u_0 = P @ X[j]
            if torch.linalg.vector_norm(u_0).round(decimals=12) != 0:
                x = X[j]
                flag = False
                break
        if flag:  # violates sign assumption, skip
            continue
        u = u.view(n)
        u_0 /= torch.abs(u @ x)
        U[:, i] = u_0
    return U


def count_sign_assumption(dataset: Iterable) -> Tuple[int, int]:
    """
    Count the number of eigenvectors violating the assumption about sign in our paper. Small graphs with no more than
    5 nodes are ignored.

    :param dataset: The dataset.
    :return: Tuple (count, total), where count is the number of eigenvectors violating the assumption and total
        is the total number of eigenvectors in the dataset.
    """
    count = 0
    total_count = 0
    for data in dataset:
        n = data.num_nodes = data.x.shape[0]
        if n <= 5:
            continue
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        A = normalize_adjacency(A)
        E, U = torch.linalg.eigh(A)
        total_count += n
        for i in range(n):
            u = U[:, i].view(n, 1)
            P = u @ u.T.view(1, n)
            E = torch.eye(n)
            J = torch.ones(n)
            Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
            Pe = [i for i in enumerate(Pe)]
            Pe.sort(key=lambda x: x[1])
            indices = [i[0] for i in Pe]
            lengths = [i[1] for i in Pe]
            _, counts = np.unique(lengths, return_counts=True)
            step = 0
            X = torch.zeros([len(counts), n])
            for j in range(len(counts)):
                for _ in range(counts[j]):
                    X[j] += E[indices[step]]
                    step += 1
                X[j] += 10 * J
            flag = True
            for j in range(len(counts)):
                u_0 = P @ X[j]
                if torch.linalg.vector_norm(u_0).round(decimals=12) != 0:
                    flag = False
                    break
            if flag:  # violates sign assumption
                count += 1
                break
    return count, total_count


def count_basis_assumption_1(dataset: Iterable) -> Tuple[int, int]:
    """
    Count the number of multiple eigenvalues violating the first assumption about basis in our paper. Small graphs
    with no more than 5 nodes are ignored.

    :param dataset: The dataset.
    :return: Tuple (count, total), where count is the number of multiple eigenvalues violating the assumption and
        total is the total number of multiple eigenvalues in the dataset.
    """
    count = total_count = 0
    for data in dataset:
        n = data.num_nodes = data.x.shape[0]
        if n <= 5:
            continue
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        A = normalize_adjacency(A)
        E, U = torch.linalg.eigh(A)
        eig_vals, multiplicities = np.unique(E.tolist(), return_counts=True)
        distincts = len(eig_vals)
        step = 0
        for i in range(distincts):
            if multiplicities[i] == 1:  # eigenvalue has multiplicity 1
                step += 1
                continue
            total_count += 1
            d = multiplicities[i]
            Ui = U[:, step:step+d]
            assert list(Ui.shape) == [n, d]
            step += d
            P = Ui @ Ui.t()
            Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
            lengths = np.unique(Pe)
            if len(lengths) < d:  # k < d, violating the first assumption about basis
                count += 1
    return count, total_count


def is_linear_independent(U: Tensor) -> bool:
    """
    Returns True if columns of U are linear independent.

    >>> U = Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
    >>> is_linear_independent(U)
    True
    >>> U = Tensor([[1, 0, 2], [0, 0, 0], [0, 1, 3], [0, 0, 0], [0, 0, 0]])
    >>> is_linear_independent(U)
    False

    :param U: Tensor of shape [n, d] (n >= d).
    :return: Boolean value.
    """
    n, d = U.shape
    rank = torch.linalg.matrix_rank(U).item()
    return d == rank


def orthogonalize(U: Tensor) -> Tensor:
    """
    Orthogonalize a set of linear independent vectors using Gram–Schmidt process.

    >>> U = torch.nn.functional.normalize(torch.randn(5, 3), dim=0)
    >>> U = orthogonalize(U)
    >>> torch.allclose(U.T @ U, torch.eye(3), atol=1e-06)
    True

    :param U: Tensor of shape [n, d], d <= n.
    :return: Tensor of shape [n, d].
    """
    Q, R = torch.linalg.qr(U)
    return Q


def random_orthonormal_matrix(n: int, d: int) -> Tensor:
    """
    Randomly generate an orthonormal matrix of shape [n, d].

    >>> U = random_orthonormal_matrix(5, 3)
    >>> I = torch.eye(3)
    >>> torch.allclose(U.T @ U, I, atol=1e-06)
    True

    :param n: The first dimension of the random orthonormal matrix.
    :param d: The second dimension of the random orthonormal matrix.
    :return: Random orthonormal matrix of shape [n, d].
    """
    A = torch.randn([n, n])
    _, U = torch.linalg.eigh(A)
    return U[:, :d]


def random_sign_matrix(n: int) -> Tensor:
    """
    Randomly generate a diagonal matrix of 1 and -1.

    :param n: The size of the matrix.
    :return: Random sign matrix of shape [n, n].
    """
    s = torch.randint(0, 2, [n])
    s = 2 * s - 1
    S = torch.diag(s.to(float))
    return S


def random_permutation_matrix(n: int) -> Tensor:
    """
    Generate a random permutation matrix.

    :param n: The order of the permutation matrix.
    :return: Tensor of shape [n, n].
    """
    P = torch.eye(n)
    sigma = torch.randperm(n)
    return P[sigma]


def find_complementary_space(U: Tensor, u_span: Tensor) -> Tensor:
    """
    Find the orthogonal complementary space of u_span in the linear space U.

    >>> U = Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    >>> u_span = Tensor([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1]])
    >>> find_complementary_space(U, u_span)
    tensor([[1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [0., 0.]])

    :param U: Tensor of shape [n, d].
    :param u_span: Tensor of shape [n, s], where s <= d.
    :return: Tensor of shape [n, d - s].
    """
    n, d = U.shape
    s = u_span.shape[1]
    u_base = u_span.clone()
    for j in range(d):
        i = u_base.shape[1]
        u_j = U[:, j].unsqueeze(dim=1)  # shape [n, 1]
        u_temp = torch.cat([u_base, u_j], dim=1)  # shape [n, d'] where i <= d' <= d
        if torch.linalg.matrix_rank(u_temp) == i + 1:  # u_temp are linear independent
            u_base = u_temp
        if u_base.shape[1] == d:
            break
    u_base = orthogonalize(u_base)
    u_perp = u_base[:, s:d]
    return u_perp


def unique_basis(U_i: Tensor) -> Tensor:
    """
    Eliminating basis ambiguity of the input eigenvectors.

    :param U_i: Tensor of shape [n, d]. Each column of U is an eigenvector.
    :return: Tensor of shape [n, d].
    """
    n, d = U_i.shape
    E = torch.eye(n)
    J = torch.ones(n)
    P = U_i @ U_i.T
    Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
    Pe = [i for i in enumerate(Pe)]
    Pe.sort(key=lambda x: x[1])
    indices = [i[0] for i in Pe]
    lengths = [i[1] for i in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    assert len(counts) >= d  # basis assumption 1
    X = torch.zeros([d, n])  # [x_1, ..., x_d]
    step = -1
    for i in range(1, d + 1):
        x = torch.zeros(n)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    U_0 = torch.zeros([n, d])  # the unique basis
    u_span = torch.empty([n, 0])  # span(u_1, ..., u_{i-1})
    u_perp = U_i.clone()  # orthogonal complementary space
    for i in range(d):
        P_perp = u_perp @ u_perp.T
        u_i = P_perp @ X[i]
        assert torch.linalg.vector_norm(u_i) != 0  # basis assumption 2
        u_i = torch.nn.functional.normalize(u_i, dim=0)
        U_0[:, i] = u_i
        u_span = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
        u_perp = find_complementary_space(U_i, u_span)
    return U_0


def count_basis_assumption_2(dataset: Iterable) -> Tuple[int, int, int]:
    """
    Count the number of multiple eigenvalues violating the second assumption about basis in our paper. Small graphs
    with no more than 5 nodes are ignored.

    :param dataset: The dataset.
    :return: Tuple (count_1, count_2, total), where count_1 is the number of multiple eigenvalues violating the first
        assumption, count_2 is the number of multiple eigenvalues that does not violate the first assumption but
        violates the second assumption, and total is the total number of multiple eigenvalues in the dataset.
    """
    count_1 = count_2 = total_count = 0
    for data in dataset:
        n = data.num_nodes = data.x.shape[0]
        if n <= 5:
            continue
        if data.edge_index.shape[1] == 0:
            A = torch.zeros([n, n])
        else:  # no edges in the graph
            A = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=n))
        A = normalize_adjacency(A)
        E, U = torch.linalg.eigh(A)
        E = E.round(decimals=14)
        U = unique_sign(U)
        J = torch.ones(n)
        _, mult = torch.unique(E, return_counts=True)
        for U_i in torch.split(U, mult.tolist(), dim=1):
            d = U_i.shape[1]
            if d == 1:
                continue
            total_count += 1
            E = torch.eye(n)
            P = U_i @ U_i.T
            Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
            Pe = [i for i in enumerate(Pe)]
            Pe.sort(key=lambda x: x[1])
            indices = [i[0] for i in Pe]
            lengths = [i[1] for i in Pe]
            _, counts = np.unique(lengths, return_counts=True)
            if len(counts) < d:  # violates basis assumption 1
                count_1 += 1
                continue
            X = torch.zeros([d, n])  # [x_1, ..., x_d]
            step = -1
            for i in range(1, d + 1):
                x = torch.zeros(n)
                for _ in range(counts[-i]):
                    x += E[indices[step]]
                    step += 1
                X[i - 1] = x + 10 * J
            U_0 = torch.zeros([n, d])  # the unique basis
            u_span = torch.empty([n, 0])  # span(u_1, ..., u_{i-1})
            u_perp = U_i.clone()  # orthogonal complementary space
            for i in range(d):
                P_perp = u_perp @ u_perp.T
                u_i = P_perp @ X[i]
                if torch.linalg.vector_norm(u_i).round(decimals=14) == 0:  # violates basis assumption 2
                    count_2 += 1
                    break
                u_i = torch.nn.functional.normalize(u_i, dim=0)
                U_0[:, i] = u_i
                u_span = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
                u_perp = find_complementary_space(U_i, u_span)
    return count_1, count_2, total_count


if __name__ == "__main__":
    # ==================== Verifying Algorithm 1 ====================

    # test permutation-equivariance of our sign algorithm
    torch.set_default_dtype(torch.float64)
    p_correct = q_correct = pq_correct = total = 0
    epochs = 1000
    for _ in range(epochs):
        n = torch.randint(2, 20, [1]).item()
        U = random_orthonormal_matrix(n, n)
        U_0 = unique_sign(U)

        # test permutation equivariance
        P = random_permutation_matrix(n)
        V = P @ U
        V_0 = unique_sign(V)
        p_correct += torch.allclose(P @ U_0, V_0, atol=1e-06)

        # test uniqueness
        S = random_sign_matrix(n)
        W = U @ S
        W_0 = unique_sign(W)
        q_correct += torch.allclose(U_0, W_0, atol=1e-06)

        # test both
        Y = P @ W
        Y_0 = unique_sign(Y)
        pq_correct += torch.allclose(P @ U_0, Y_0, atol=1e-06)

        total += 1
    print("Test results for Algorithm 1:")
    print(f"Permutation-equivariance: {p_correct} / {total}")
    print(f"Uniqueness: {q_correct} / {total}")
    print(f"Both at the same time: {pq_correct} / {total}")
    print()
    # Output:
    # Permutation-equivariance: 1000 / 1000
    # Uniqueness: 1000 / 1000
    # Both at the same time: 1000 / 1000

    # ==================== Verifying Algorithm 2 ====================

    # test the uniqueness and permutation-equivariance of our basis algorithm
    p_correct = q_correct = pq_correct = total = 0
    epochs = 1000
    for _ in range(epochs):
        n = torch.randint(2, 20, [1]).item()
        d = torch.randint(1, n, [1]).item()
        U = random_orthonormal_matrix(n, d)
        try:
            U_0 = unique_basis(U)
        except AssertionError:  # assumptions not satisfied, skip
            continue

        # test permutation equivariance
        P = random_permutation_matrix(n)
        V = P @ U
        try:
            V_0 = unique_basis(V)
            flag_p = torch.allclose(P @ U_0, V_0, atol=1e-06)
        except AssertionError:  # assumptions not satisfied, skip
            continue

        # test uniqueness
        Q = random_orthonormal_matrix(d, d)
        W = U @ Q
        try:
            W_0 = unique_basis(W)
            flag_q = torch.allclose(U_0, W_0, atol=1e-06)
        except AssertionError:  # assumptions not satisfied, skip
            continue

        # test both
        Y = P @ W
        try:
            Y_0 = unique_basis(Y)
            flag_pq = torch.allclose(P @ U_0, Y_0, atol=1e-06)
        except AssertionError:  # assumptions not satisfied, skip
            continue

        total += 1
        p_correct += flag_p
        q_correct += flag_q
        pq_correct += flag_pq
    print("Test results for Algorithm 2:")
    print(f"Permutation-equivariance: {p_correct} / {total}")
    print(f"Uniqueness: {q_correct} / {total}")
    print(f"Both at the same time: {pq_correct} / {total}")
    print(f"Assumptions are almost never violated: {total} / {epochs}")
    # output:
    # Permutation-equivariance: 1000 / 1000
    # Uniqueness: 1000 / 1000
    # Both at the same time: 1000 / 1000
    # Assumptions are almost never violated: 1000 / 1000
