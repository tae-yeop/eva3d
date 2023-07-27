from pytorch_fid import fid_score

def get_fid(our_path, target_path):
    return fid_score.calculate_fid_given_paths(
        [our_path, target_path],
        50, 'cuda', 2048, 8
    )


import configargparse
from munch import *
if __name__ == '__main__':
    opt = Munch()
    
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--our_path", type=str, default='/home/aiteam/tykim/generative_model/human/EVA3D/evaluations/512x256_deepfashion/iter_0420000/random_angles/images_paper_fig')
    
    parser.add_argument("--target_path", type=str)

    args = parser.parse_args()

    # for group in parser._action_groups[2:]:
    #     title = group.title
    #     opt[title] = Munch()
    #     for action in group._group_actions:
    #         dest = action.dest
    #         opt[title][dest] = args.__getattribute__(dest)

                
    print(get_fid(args.our_path, args.target_path))