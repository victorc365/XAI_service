import sys 
import os 
import pytest
from pathlib import Path

# add modules to system path 
sys.path.insert(0, os.path.join(__file__, '..'))
# import testing modules 

from recommender_service import RecommenderService

@pytest.fixture
def recommender_service():
    model_path = Path(__file__).parent / 'test_model'
    return RecommenderService(model_path)

