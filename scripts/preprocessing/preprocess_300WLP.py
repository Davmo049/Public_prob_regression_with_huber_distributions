import DS_300W_LP.Dataset

def main():
    ds = DS_300W_LP.Dataset.Dataset_300W_LP_Raw()
    ds.preprocess_dataset()

if __name__ == '__main__':
    main()
