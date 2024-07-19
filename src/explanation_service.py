import dill
import pickle
import bnlearn as bn
from typing import Any, Union, List
from dexire_pro.core.clustering_explanations import ClusteringExplanationGenerator

class ExplanationService:
    def __init__(self):
        self.rule_set = None
        self.bn_model = None
        self.cluster_model = None
        
    def load_cluster_model(self, cluster_model_path: str) -> None:
        self.cluster_model = ClusteringExplanationGenerator()
        self.cluster_model.load_cluster_model(cluster_model_path)
    
    def load_rule_set(self, rule_set_path: str) -> None:
        with open(rule_set_path, 'rb') as file:
            self.rule_set = dill.load(file)
        print(f"Rule set loaded successfully!")
            
    def load_bn_model(self, bn_model_path: str) -> None:
        self.bn_model = bn.load(bn_model_path)
        print(f"BN model loaded successfully!")
        
    def data_preprocessing_for_rules(self, data: dict, transformer: Any) -> dict:
        if transformer is not None:
            pass
        else:
            return data
    
    def data_preprocessing_for_bn(self, data: dict, transformer: Any) -> dict:
        if transformer is not None:
            pass
        else:
            return data
    
    def explain_decision_with_rules(self, data: Any) -> Union[List[str], str]:
        pass
    
    def explanation_bayesian_network(self, data: Any) -> Union[List[str]]:
        pass
    
    
    
if __name__ == "__main__":
    explanation_service = ExplanationService()
    explanation_service.load_rule_set('model_assets\Full_model_bert.pkl')
    explanation_service.load_bn_model('model_assets/bn_model_bert.pkl')