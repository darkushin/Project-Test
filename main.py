import torch
import Network
import Network_res3d
from data_handler import *

parser = utils.create_parser()
args = parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    # read json return
    config = utils.startup(json_path=args.config, args=args, copy_files=args.eval is None or args.eval == 'empty')

    # get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {'batch_size': config["batch_size"],
              'shuffle': True,
              'num_workers': 0,
              'worker_init_fn': worker_init_fn}
    cur_data_path = './Example-Data/billiard'

    output = None
    scale = 2

    dataset = DataHandler(data_path=cur_data_path, config=config, upsample_scale=scale, device=device)
    assert dataset.crop_size[0] % scale == 0, f'assertion error in main, temporal crop size not divisible by scale'
    data_generator = data.DataLoader(dataset, **params)

    network_class = config['network']
    if network_class == 'base':
        network = Network.Network(config=config, device=device, upsample_scale=scale)
    elif network_class == 'residual':
        network = Network_res3d.Network_residual(config=config, device=device, upsample_scale=scale)
    else:
        assert False, f'assertion fail at main, not a known "network_class"'

    # call train - provide a data_handler object to provide (lr,hr) tuples
    network.train(data_generator, scale)

    # save final result in "output" folder
    final_output_dir = os.path.join(config['trainer']['working_dir'], 'output')
    utils.save_output_result(output, final_output_dir)


if __name__ == '__main__':
    main()
    print('done.')
