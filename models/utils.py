from dhg import Hypergraph


def data2hg(data):
    num_v = len(data.x)
    e_list = []
    for i in data.edge_index1.unique().tolist():
        _e = data.edge_index0[data.edge_index1 == i].tolist()
        e_list.append(_e)

    hg = Hypergraph(num_v=num_v, e_list=e_list, device=data.x.device)
    return hg
