from simulation import run_simulation
import numpy as np
# This import is necessary for pickle to load the model
from predictions import PositionSplitEstimator


def main():
    
    total_points_1 = run_simulation('2023-24', log=False)
    print(f"Total points for 2023-24: {total_points_1}")
    total_points_2 = run_simulation('2022-23', log=False)
    print(f"Total points for 2022-23: {total_points_2}")
    total_points_3 = run_simulation('2021-22', log=False)
    print(f"Total points for 2021-22: {total_points_3}")
    average = np.mean([total_points_1, total_points_2, total_points_3])
    print(f"Average total points: {average:.2f}")

if __name__ == "__main__":
    main()