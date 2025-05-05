from uniform_prefill import Engine as UniformEngine
from different_prefill import Engine as DifferentEngine
from single_batch import Engine as SingleEngine

BASE_PROMPT = "Hi, who are you?"

def test_deterministic():
    """
    Test the deterministic behavior of the model.
    """
    input_string_list = [BASE_PROMPT] 
    
    # Initialize engines
    uniform_engine = UniformEngine()
    different_engine = DifferentEngine()
    single_engine = SingleEngine()

    # Generate text using different engines
    uniform_output = uniform_engine.generate_batched(input_string_list, rounds=200)
    different_output = different_engine.generate_batched(input_string_list, rounds=200)
    single_output = single_engine.generate(BASE_PROMPT, rounds=200)

    # Check if outputs are deterministic
    assert uniform_output == different_output == [single_output], "Outputs are not deterministic!"
    
    
if __name__ == "__main__":  
    test_deterministic()
    print("All tests passed!")