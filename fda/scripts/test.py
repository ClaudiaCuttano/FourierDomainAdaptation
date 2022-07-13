import sys
sys.path.append('/home/ccuttano/FDA')

import argparse
import pprint
import warnings
from fda.domain_adaptation.config import cfg, cfg_from_file
from fda.domain_adaptation.eval_UDA import eval_single
from fda.domain_adaptation.evaluation_multi import eval_multi

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    if cfg.TEST.MULTI ==True:
        eval_multi(cfg)
    else:
        eval_single(cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg)
