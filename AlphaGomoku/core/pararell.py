import multiprocessing as mp
import AlphaGomoku.core.run_gomoku_2

for a in range(3):
    t = mp.Process(target=run_gomoku2)
    t.start()
    print("Thread started")
    sleep(0.5)
