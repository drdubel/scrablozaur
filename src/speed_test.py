from multiprocessing import Pool

from tqdm import tqdm

from main import play_game


def test():
    n = 1000
    points = 0
    with Pool(processes=25) as pool:
        for score1, score2 in tqdm(pool.imap_unordered(play_game, range(n)), total=n):
            points += score1 + score2
    print(points / n)


if __name__ == "__main__":
    test()
