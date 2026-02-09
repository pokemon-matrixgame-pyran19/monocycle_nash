from dataclasses import dataclass
from domain.model import DomainModel

# このファイル自体は実際にうごかさない
# domainやmodelはテスト対象にあわせてさし替える

@dataclass(frozen=True)
class ScenarioBase:
    x: float
    y: float

    def domain(self) -> DomainModel:
        return DomainModel(x=self.x, y=self.y)
