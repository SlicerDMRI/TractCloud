"""Reference from https://github.com/zhangfanmark/DeepWMA"""
import numpy as np
import whitematteranalysis as wma
import sys

sys.path.append('..')

def feat_RAS(pd_tract, number_of_points=15):
    """The most simple feature for initial test"""

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_tract, points_per_fiber=number_of_points)
    # fiber_array_r, fiber_array_a, fiber_array_s have the same size: [number of fibers, points of each fiber]
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))

    return feat, fiber_array