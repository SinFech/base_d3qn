# %%
from prettytable import PrettyTable as PrettyTable
from utils import print_stats, plot_multiple_conf_interval
import random
import os
#from google.colab import drive
#drive.mount('/content/drive')
from Environment import Environment
from Agent import Agent
import pandas as pd

# %%
import pandas as pd

# Load the dataset
df = pd.read_csv("Bitcoin History 2010-2024.csv")

# Rename the 'Price' column to 'Close' and remove commas from the 'Close' values, then convert to float
df = df.rename(columns={"Price": "Close"})
df['Close'] = df['Close'].str.replace(',', '').astype(float)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame to only include rows within the specified start and end dates
start_date = '2016-07-09'
end_date = '2018-01-09'
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
df_filtered2 = df.loc[mask]

# Reset the index of the filtered DataFrame
df_filtered2 = df_filtered2.reset_index(drop=True)

df_filtered2 = df_filtered2.reset_index(drop=True)

df_filtered2 = df_filtered2.sort_values(by='Date')

# Reset the index of the filtered and sorted DataFrame
df_filtered2 = df_filtered2.reset_index(drop=True)
df_filtered2

# %%
from prettytable import PrettyTable
from utils import print_stats, plot_multiple_conf_interval
import random
import warnings
from Environment import Environment
from Agent import Agent


def main():
    # ----------------------------- LOAD DATA ---------------------------------------------------------------------------
    path = ''
    df = df_filtered2  # 你要确保 df_filtered2 已经在外部定义或加载

    # ----------------------------- PARAMETERS --------------------------------
    REPLAY_MEM_SIZE = 10000
    BATCH_SIZE = 40
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_STEPS = 200
    LEARNING_RATE = 0.0005
    INPUT_DIM = 24
    HIDDEN_DIM = 120
    ACTION_NUMBER = 3
    TARGET_UPDATE = 5
    N_TEST = 10
    TRADING_PERIOD = 500

    # ----------------------------- ENVIRONMENTS --------------------------------
    index = random.randrange(len(df) - TRADING_PERIOD - 1)
    train_size = int(TRADING_PERIOD * 0.8)

    profit_dueling_ddqn_return = []
    sharpe_dueling_ddqn_return = []

    # ----------------------------- CREATE D3QN AGENT --------------------------------
    d3qn_agent = Agent(REPLAY_MEM_SIZE,
                       BATCH_SIZE,
                       GAMMA,
                       EPS_START,
                       EPS_END,
                       EPS_STEPS,
                       LEARNING_RATE,
                       INPUT_DIM,
                       HIDDEN_DIM,
                       ACTION_NUMBER,
                       TARGET_UPDATE,
                       MODEL='ddqn',
                       DOUBLE=True)

    # ----------------------------- TRAIN (if needed) --------------------------------
    # 如果你已经有训练好的模型，可以跳过训练部分
    # profit_train_env = Environment(df[index:index + train_size], "profit")
    # sharpe_train_env = Environment(df[index:index + train_size], "sr")
    #
    # d3qn_agent.train(profit_train_env, path)
    # profit_train_env.reset()
    #
    # d3qn_agent.train(sharpe_train_env, path)
    # sharpe_train_env.reset()

    # ----------------------------- TEST PROFIT --------------------------------
    for i in range(N_TEST):
        print(f"Profit Test {i + 1}")
        index = random.randrange(len(df) - TRADING_PERIOD - 1)
        profit_test_env = Environment(df[index + train_size:index + TRADING_PERIOD], "profit")

        cr_profit_test, _ = d3qn_agent.test(
            profit_test_env,
            model_name="profit_reward_double_ddqn_model",
            path=path
        )
        profit_dueling_ddqn_return.append(profit_test_env.cumulative_return)
        profit_test_env.reset()

    # ----------------------------- TEST SHARPE --------------------------------
    for i in range(N_TEST):
        print(f"Sharpe Test {i + 1}")
        index = random.randrange(len(df) - TRADING_PERIOD - 1)
        sharpe_test_env = Environment(df[index + train_size:index + TRADING_PERIOD], "sr")

        cr_sharpe_test, _ = d3qn_agent.test(
            sharpe_test_env,
            model_name="sr_reward_double_ddqn_model",
            path=path
        )
        sharpe_dueling_ddqn_return.append(sharpe_test_env.cumulative_return)
        sharpe_test_env.reset()

    # ----------------------------- PRINT STATS --------------------------------
    t = PrettyTable(["Trading System", "Avg. Return (%)", "Max Return (%)", "Min Return (%)", "Std. Dev."])
    print_stats("Profit D3QN", profit_dueling_ddqn_return, t)
    print_stats("Sharpe D3QN", sharpe_dueling_ddqn_return, t)
    print(t)

    # ----------------------------- PLOT --------------------------------
    plot_multiple_conf_interval(
        ["Profit D3QN", "Sharpe D3QN"],
        [profit_dueling_ddqn_return, sharpe_dueling_ddqn_return]
    )


if __name__ == "__main__":
    main()



