import random
import matplotlib.pyplot as plt

ARMS = 5
TIMESTEPS = 100
RUNS = 1000
EPSILON = 0.1


class Q_value:
    def __init__(self) -> None:
        self.q = 0
        self.n = 0
        
    def update(self, r) -> None:
        self.n += 1
        self.q = self.q + ((1/self.n) * (r - self.q))

class Q_table:
    def __init__(self) -> None:
        self.table = []
        self.initialize_table()
    
    def initialize_table(self) -> None:
        for t in range(TIMESTEPS):
            i = 0
            step_table = []
            while i < ARMS:
               new_q = Q_value()
               step_table.append(new_q)
               i += 1
               
            self.table.append(step_table)
    
    def find_max_q(self, row:int) -> int:
        q_values = []
        for i in self.table[row]:
            q_values.append(i.q)
        index_max = max(range(len(q_values)), key=q_values.__getitem__)
        return index_max
    
    def update_table(self, row:int, column:int, value:float) -> None:
        self.table[row][column].update(value)
        

class Bandit:
    """This class represents the n-armed bandit 
    """
    def __init__(self, arms, timesteps, runs, epsilon)->None:
        self.arms = arms
        self.timesteps = timesteps
        self.runs = runs
        self.epsilon = epsilon
        self.distribution = []
        self.q_table = Q_table()
        self.greedy_table = Q_table()
        self.create_distribution()
    
    def create_distribution(self) -> None:
        """This function creates the array of random reward values
        """        
        for t in range(TIMESTEPS):
            i = 0
            step_dist = []
            while i < ARMS:
                a = random.random()
                step_dist.append(round(a,1))
                i += 1
            self.distribution.append(step_dist)
            
    def epsilon_greedy(self) -> bool:
        """This function determines whether a greedy action or a random action will be taken

        Returns:
            bool: Returns True for epsilon greedy action
        """
        dice_roll = random.random()
        
        # dice_roll generates a random float between 0 and 1. If that is less than the value of
        # epsilon, then we make a random move, otherwise we make a greedy move.
        if dice_roll <= EPSILON:
            return False
        return True
    
    def run_bandit(self) -> float:
        """This function runs the n-armed bandit one time, and updates the q-table based on the result

        Returns:
            float: Returns the average reward value of all timesteps
        """
        reward_values = []
        step = 0
        while step < TIMESTEPS:
            if self.epsilon_greedy():
                index_max = self.q_table.find_max_q(row=step)
                reward = self.distribution[step][index_max]
                self.q_table.update_table(row=step, column=index_max, value=reward)
                reward_values.append(reward)
                
            else:
                random_action = random.randint(0,(ARMS-1))
                reward = self.distribution[step][random_action]
                self.q_table.update_table(row=step, column=random_action, value=reward)
                reward_values.append(reward)

            step += 1
        
        average_reward = sum(reward_values)/TIMESTEPS
        return average_reward 
    
    def run_greedy_bandit(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        reward_values = []
        step = 0
        while step < TIMESTEPS:
            index_max = self.greedy_table.find_max_q(row=step)
            reward = self.distribution[step][index_max]
            self.greedy_table.update_table(row=step, column=index_max, value=reward)
            reward_values.append(reward)
        
            step += 1
        
        average_reward = sum(reward_values)/TIMESTEPS
        return average_reward 

bandit = Bandit(ARMS, TIMESTEPS, RUNS, EPSILON)

for i in bandit.distribution:
    print(i)

epsilon_averages = []
greedy_averages = []
run_count = []
for run in range(RUNS):
    run_average = bandit.run_bandit()
    greedy_average = bandit.run_greedy_bandit()
    epsilon_averages.append(run_average)
    greedy_averages.append(greedy_average)
    run_count.append(run)

plt.plot(run_count, epsilon_averages, label="Epsilon-Greedy")
plt.plot(run_count, greedy_averages, label="Greedy")
plt.legend()
plt.title(f"{ARMS}-Armed Bandit")
plt.xlabel("Runs")
plt.ylabel("Average Reward")
plt.show()


