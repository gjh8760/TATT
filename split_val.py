import os
import lmdb
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, default='/data/gjh8760/Dataset/STISR/TextZoom/test/medium', help='')
    parser.add_argument('--new_lmdb_path', type=str, default='/data/gjh8760/Dataset/STISR/TextZoom/test/medium_val', help='')
    args = parser.parse_args()

    lmdb_path = args.lmdb_path
    new_lmdb_path = args.new_lmdb_path
    if not os.path.exists(new_lmdb_path):
        os.makedirs(new_lmdb_path)

    env = lmdb.open(lmdb_path, map_size=1099511627776)
    txn = env.begin()

    num_samples = int(txn.get(b'num-samples'))
    # 10n 번째 sample을 validation set으로 사용
    env_new = lmdb.open(new_lmdb_path, map_size=1099511627776)
    txn_new = env_new.begin(write=True)
    num_samples_new = 0
    for n in range(0, num_samples-1, 10):
        image_hr_key = b'image_hr-%09d' % (n + 1)
        image_lr_key = b'image_lr-%09d' % (n + 1)
        label_key = b'label-%09d' % (n + 1)

        imgbuf_hr = txn.get(image_hr_key)
        imgbuf_lr = txn.get(image_lr_key)
        labelbuf = txn.get(label_key)

        txn_new.put(key=image_hr_key, value=imgbuf_hr)
        txn_new.put(key=image_lr_key, value=imgbuf_lr)
        txn_new.put(key=label_key, value=labelbuf)
        num_samples_new += 1

    txn_new.put(key=b'num-samples', value=str(num_samples_new).encode())

    txn_new.commit()
    env_new.close()


if __name__ == '__main__':
    main()


