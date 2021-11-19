import matplotlib.pyplot as plt
import DS_300W_LP.Dataset as Dataset

def main():
    ds = Dataset.Preprocessed300WLPKeypointsDataset()
    tds = ds.get_train()
    print('a')
    for im, points, kp_type, weights in tds:
        im = im.transpose(1,2,0)
        plt.imshow(im)
        print(points.shape)
        plt.plot(points[:,0], points[:,1], 'x')
        plt.show()

    

if __name__ == '__main__':
    main()
