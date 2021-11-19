import WFLW.WFLW as wflw
import matplotlib.pyplot as plt

def main():
    ds = wflw.PreprocessedWflwfKeypointsDataset()
    train = ds.get_train()
    im, ann, weights, point_type = train[0]

    im = im.transpose(1,2,0)
    for i in range(98):
        plt.imshow(im)
        plt.plot(ann[:, 0], ann[:, 1], 'rx')
        start_x = ann[i, 0]
        start_y = ann[i, 1]
        new_i = wflw.FLIP_ORDER[i]
        if i == new_i:
            plt.plot([start_x], [start_y], 'bo')
        else:
            end_x = ann[new_i, 0]
            end_y = ann[new_i, 1]
            plt.plot([start_x, end_x], [start_y, end_y])
        plt.show()

if __name__ == '__main__':
    main()
