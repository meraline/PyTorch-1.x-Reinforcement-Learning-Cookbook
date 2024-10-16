'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''

import gymnasium as  gym   
import numpy as np
import ale_py

# Регистрация среды Atari (ALE) в gymnasium
gym.register_envs(ale_py)

# Создание среды SpaceInvaders с отображением в режиме "human"
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", full_action_space=True)

# Обнуление среды и получение начального наблюдения
observation, info = env.reset()

# Информация о пространстве действий и наблюдений
print("Action space:", env.action_space)  # Ожидается Discrete(6)
print("Observation space:", env.observation_space)  # Ожидается Box(0, 255, (210, 160, 3), np.uint8)

# Определение основных параметров игры
num_episodes = 1  # количество эпизодов игры
episode_reward = 0  # общий счёт

# Основной игровой цикл
for episode in range(num_episodes):
    done = False
    observation, info = env.reset()
    episode_reward = 0
    
    while not done:
        # Рендеринг кадра
        env.render()
        
        # Случайное действие из пространства действий Discrete(6)
        action = env.action_space.sample()
        
        # Шаг в среде
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Суммирование общего счёта за эпизод
        episode_reward += reward
        
        # Проверка на завершение игры
        done = terminated or truncated
    
    # Вывод общего счёта за эпизод
    print(f"Episode {episode + 1} completed with total reward: {episode_reward}")

# Закрытие среды
env.close()



