from .registry import StrategyRegistry

if __name__ == "__main__":
    print("strategy package self-test")
    reg = StrategyRegistry()
    print("Available strategies:", ["abu_key_level"])
