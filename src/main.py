from scrablozaur import Board, Dawg  # type: ignore

d = Dawg("words/dawg.bin")
b = Board([["-" for _ in range(15)] for _ in range(15)])

print(b.calculate_word_points("odholował", 7, 1, True))