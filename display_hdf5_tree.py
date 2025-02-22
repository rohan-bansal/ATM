import h5py

filename_hdf = '/home/terra/dev/Research/rl2/atm/diffusion_policy/pusht_dataset/demo_10.hdf5'

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                    if key == "waypoints_dp":
                        print(val[:])
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                    if key == "precisions":
                        print(val[:])
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')

with h5py.File(filename_hdf, 'r') as hf:
    print(hf)
    h5_tree(hf)