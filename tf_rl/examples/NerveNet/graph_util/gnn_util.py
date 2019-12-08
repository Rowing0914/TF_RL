#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Some helper functions to parse the mujoco xml template files
#   @author:
#       Tingwu Wang, Jun 23rd, 2017
#   @UPDATE:
# -----------------------------------------------------------------------------


import numpy as np

import graph_util.init_path as init_path
import graph_util.mujoco_parser as mujoco_parser
from util import logger

__all__ = ['io_size_check']
_BASE_PATH = init_path.get_abs_base_dir()


def io_size_check(input_size, output_size, node_info, is_baseline):
    '''
        @brief:
            check if the environment's input size and output size is matched
            with the one parsed from the mujoco xml file
    '''
    if not is_baseline:
        is_io_matched = \
            input_size == node_info['debug_info']['ob_size'] and \
            output_size == node_info['debug_info']['action_size']
    else:
        is_io_matched = \
            input_size == node_info['debug_info']['ob_size'] and \
            output_size == 1

    assert is_io_matched, logger.error(
        'The output and input size is not matched!' +
        ' ({}, {}) vs. ({}, {})'.format(
            input_size,
            output_size,
            node_info['debug_info']['ob_size'],
            node_info['debug_info']['action_size']
        )
    )


def construct_ob_size_dict(node_info, input_feat_dim):
    '''
        @brief: for each node type, we collect the ob size for this type
    '''
    node_info['ob_size_dict'] = {}
    for node_type in node_info['node_type_dict']:
        node_ids = node_info['node_type_dict'][node_type]

        # record the ob_size for each type of node
        if node_ids[0] in node_info['input_dict']:
            node_info['ob_size_dict'][node_type] = \
                len(node_info['input_dict'][node_ids[0]])
        else:
            node_info['ob_size_dict'][node_type] = 0

        node_ob_size = [
            len(node_info['input_dict'][node_id])
            for node_id in node_ids if node_id in node_info['input_dict']
        ]

        if len(node_ob_size) == 0:
            continue

        assert node_ob_size.count(node_ob_size[0]) == len(node_ob_size), \
            logger.error('Nodes (type {}) have wrong ob size: {}!'.format(
                node_type, node_ob_size
            ))

    return node_info


def get_inverse_type_offset(node_info, mode):
    assert mode in ['output', 'node'], logger.error(
        'Invalid mode: {}'.format(mode)
    )
    node_info['inverse_' + mode + '_extype_offset'] = []
    node_info['inverse_' + mode + '_intype_offset'] = []
    node_info['inverse_' + mode + '_self_offset'] = []
    node_info['inverse_' + mode + '_original_id'] = []
    current_offset = 0
    for mode_type in node_info[mode + '_type_dict']:
        i_length = len(node_info[mode + '_type_dict'][mode_type])
        # the original id
        node_info['inverse_' + mode + '_original_id'].extend(
            node_info[mode + '_type_dict'][mode_type]
        )

        # In one batch, how many element is listed before this type?
        # e.g.: [A, A, C, B, C, A], with order [A, B, C] --> [0, 0, 4, 3, 4, 0]
        node_info['inverse_' + mode + '_extype_offset'].extend(
            [current_offset] * i_length
        )

        # In current type, what is the position of this node?
        # e.g.: [A, A, C, B, C, A] --> [0, 1, 0, 0, 1, 2]
        node_info['inverse_' + mode + '_intype_offset'].extend(
            range(i_length)
        )

        # how many nodes are in this type?
        # e.g.: [A, A, C, B, C, A] --> [3, 3, 2, 1, 2, 3]
        node_info['inverse_' + mode + '_self_offset'].extend(
            [i_length] * i_length
        )
        current_offset += i_length

    sorted_id = np.array(node_info['inverse_' + mode + '_original_id'])
    sorted_id.sort()
    node_info['inverse_' + mode + '_original_id'] = [
        node_info['inverse_' + mode + '_original_id'].index(i_node)
        for i_node in sorted_id
    ]

    node_info['inverse_' + mode + '_extype_offset'] = np.array(
        [node_info['inverse_' + mode + '_extype_offset'][i_node]
         for i_node in node_info['inverse_' + mode + '_original_id']]
    )
    node_info['inverse_' + mode + '_intype_offset'] = np.array(
        [node_info['inverse_' + mode + '_intype_offset'][i_node]
         for i_node in node_info['inverse_' + mode + '_original_id']]
    )
    node_info['inverse_' + mode + '_self_offset'] = np.array(
        [node_info['inverse_' + mode + '_self_offset'][i_node]
         for i_node in node_info['inverse_' + mode + '_original_id']]
    )

    return node_info


def get_adjacency_matrices(node_info):
    """ Convert the relation matrices into the adjacency matrices """
    adjacency_matrices = {
        edge_type: np.array(node_info["relation_matrix"] == edge_type).astype(np.float32)
        for edge_type in node_info["edge_type_list"]
    }
    node_info["adjacency_matrix"] = adjacency_matrices
    return node_info


def get_stacked_node_params(node_info):
    """
    Returns the stacked node parameters which used for the propagation
    via adjacency matrix in the model
    """
    temp = np.array([])
    for node_type in node_info["node_type_dict"]:
        if node_type == "root":
            temp = node_info["node_parameters"][node_type]
        temp = np.vstack([temp, node_info["node_parameters"][node_type]])
    node_info["stacked_node_params"] = temp
    return node_info


def get_receive_send_idx(node_info):
    # register the edges that shows up, get the number of edge type
    edge_dict = mujoco_parser.EDGE_TYPE
    edge_type_list = []  # if one type of edge exist, register

    for edge_id in edge_dict.values():
        if edge_id == 0:
            continue  # the self loop is not considered here
        if (node_info['relation_matrix'] == edge_id).any():
            edge_type_list.append(edge_id)

    node_info['edge_type_list'] = edge_type_list
    node_info['num_edge_type'] = len(edge_type_list)

    receive_idx_raw = {}
    receive_idx = []
    send_idx = {}
    for edge_type in node_info['edge_type_list']:
        receive_idx_raw[edge_type] = []
        send_idx[edge_type] = []
        i_id = np.where(node_info['relation_matrix'] == edge_type)
        for i_edge in range(len(i_id[0])):
            send_idx[edge_type].append(i_id[0][i_edge])
            receive_idx_raw[edge_type].append(i_id[1][i_edge])
            receive_idx.append(i_id[1][i_edge])

    node_info['receive_idx'] = receive_idx
    node_info['receive_idx_raw'] = receive_idx_raw
    node_info['send_idx'] = send_idx
    node_info['num_edges'] = len(receive_idx)

    return node_info

def add_node_info(node_info, input_feat_dim):
    # step 2: check for ob size for each node type, construct the node dict
    node_info = construct_ob_size_dict(node_info, input_feat_dim)

    # step 3: get the inverse node offsets (used to construct gather idx)
    node_info = get_inverse_type_offset(node_info, 'node')

    # step 4: get the inverse node offsets (used to gather output idx)
    node_info = get_inverse_type_offset(node_info, 'output')

    # step 5: register existing edge and get the receive and send index
    node_info = get_receive_send_idx(node_info)

    # step 6: get the stacked node params
    node_info = get_stacked_node_params(node_info)

    # step 7: get the adjacency matrices
    node_info = get_adjacency_matrices(node_info)
    return node_info