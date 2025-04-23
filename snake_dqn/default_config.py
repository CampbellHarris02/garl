default_config = {
    "lr": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,          # Add this
    "epsilon_decay": 0.995,
    "epsilon_min": 0.001,
    "batch_size": 64,
    "replay_buffer_size": 2000,
    "target_update_interval": 5,
    "hidden1": 128,
    "hidden2": 64,
}
