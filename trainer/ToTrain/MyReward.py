from ASRCAISim1.libCore import TeamReward, nljson, Fighter

import sys
class MyReward(TeamReward):
    """
    チーム全体で共有する報酬は TeamReward を継承し、
    個別の Agent に与える報酬は AgentReward を継承する。
    """
    def __init__(self, modelConfig: nljson, instanceConfig: nljson):
        super().__init__(modelConfig, instanceConfig)
        if(self.isDummy):
            return #Factory によるダミー生成のために空引数でのインスタンス化に対応させる


    def onEpisodeBegin(self):
        """
        エピソード開始時の処理(必要に応じてオーバーライド)
        基底クラスにおいて config に基づき報酬計算対象の設定等が行われるため、
        それ以外の追加処理や設定の上書きを行いたい場合のみオーバーライドする。
        """
        super().onEpisodeBegin()


    def onStepBegin(self):
        """
        step 開始時の処理(必要に応じてオーバーライド)
        基底クラスにおいて reward(step 報酬)を 0 にリセットしているため、
        オーバーライドする場合、基底クラスの処理を呼び出すか、同等の処理が必要。
        """
        super().onEpisodeBegin()
    
    
    def onInnerStepBegin(self):
        """
        インナーステップ開始時の処理(必要に応じてオーバーライド)
        デフォルトでは何も行わないが、より細かい報酬計算が必要な場合に使用可能。
        """
        pass
    
    
    def onInnerStepEnd(self):
        """
        インナーステップ終了時の処理(必要に応じてオーバーライド)
        一定周期で呼び出されるため、極力この関数で計算する方が望ましい。
        """
        # スコアだけ見てるやつ
        teams = self.reward
        teams = list(teams.keys())
        for ti,team in enumerate(teams): # team に属している Asset(Fighter)を取得する例
            for f in self.manager.getAssets(lambda a:a.getTeam()==team and isinstance(a,Fighter)):
                self.reward[team] += self.manager.scores[team]*0.1
                for other in [ot for oti,ot in enumerate(teams) if oti != ti]:
                    for of in self.manager.getAssets(lambda b:b.getTeam()==other and isinstance(b,Fighter)):
                        self.reward[team] -= self.manager.scores[other]*0.1
                
