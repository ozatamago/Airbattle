import numpy as np
from gymnasium import spaces
from math import cos, sin, atan2, sqrt
from ASRCAISim1.libCore import Agent, MotionState, Track3D, Track2D, getValueFromJsonKRD, deg2rad, StaticCollisionAvoider2D, LinearSegment, AltitudeKeeper
from .scripts.Core import DataNormalizer

class MyAgent(Agent):

    class TeamOrigin():
        #陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス。
        #MotionStateを使用しても良いがクォータニオンを経由することで浮動小数点演算に起因する余分な誤差が生じるため、もし可能な限り対称性を求めるのであればこの例のように符号反転で済ませたほうが良い。
        #ただし、機体運動等も含めると全ての状態量に対して厳密に対称なシミュレーションとはならないため、ある程度の誤差は生じる。
        def __init__(self,isEastSider_,dLine):
            self.isEastSider=isEastSider_
            if(self.isEastSider):
                self.pos=np.array([0.,dLine,0.])
            else:
                self.pos=np.array([0.,-dLine,0.])
        def relBtoP(self,v):
            #陣営座標系⇛慣性座標系
            if(self.isEastSider):
                return np.array([v[1],-v[0],v[2]])
            else:
                return np.array([-v[1],v[0],v[2]])
        def relPtoB(self,v):
            #慣性座標系⇛陣営座標系
            if(self.isEastSider):
                return np.array([-v[1],v[0],v[2]])
            else:
                return np.array([v[1],-v[0],v[2]])


    class ActionInfo():
        def __init__(self):
            self.dstDir=np.array([1.0,0.0,0.0]) #目標進行方向
            self.dstAlt=10000.0 #目標高度
            self.velRecovery=False #下限速度制限からの回復中かどうか
            self.asThrottle=False #加減速についてスロットルでコマンドを生成するかどうか
            self.keepVel=False #加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
            self.dstThrottle=1.0 #目標スロットル
            self.dstV=300.0 #目標速度
            self.launchFlag=False #射撃するかどうか
            self.target=Track3D() #射撃対象
            self.lastShotTimes={} #各Trackに対する直前の射撃時刻


    def __init__(self,modelConfig,instanceConfig):
        super().__init__(modelConfig,instanceConfig) # 設定の読み込み
        if(self.isDummy):
            return # Factoryによるダミー生成のために空引数でのインスタンス化に対応させる
        self.normalizer = DataNormalizer(modelConfig,instanceConfig)
        self.time_params = self.modelConfig['interval']()
        self.to_list = self.modelConfig['to_list']()
        self.own = self.getTeam()
        self.maxTrackNum=getValueFromJsonKRD(self.modelConfig,"maxTrackNum",self.randomGen,{"Friend":4,"Enemy":4})
        self.last_action_dim=3+(1+self.maxTrackNum["Enemy"])
        self.maxMissileNum=getValueFromJsonKRD(self.modelConfig,"maxMissileNum",self.randomGen,{"Friend":8,"Enemy":1})
        # 場外制限に関する設定
        self.dOutLimit=getValueFromJsonKRD(self.modelConfig,"dOutLimit",self.randomGen,5000.0)
        self.dOutLimitThreshold=getValueFromJsonKRD(self.modelConfig,"dOutLimitThreshold",self.randomGen,10000.0)
        self.dOutLimitStrength=getValueFromJsonKRD(self.modelConfig,"dOutLimitStrength",self.randomGen,2e-3)
        
        # 左右旋回に関する設定
        self.turnTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"turnTable",self.randomGen,
			[-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0])),dtype=np.float64)
        self.turnTable*=deg2rad(1.0)
        self.use_override_evasion=getValueFromJsonKRD(self.modelConfig,"use_override_evasion",self.randomGen,True)
        if(self.use_override_evasion):
            self.evasion_turnTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"evasion_turnTable",self.randomGen,
                    [-90.0,-45.0,-20.0,-10.0,0.0,10.0,20.0,45.0,90.0])),dtype=np.float64)
            self.evasion_turnTable*=deg2rad(1.0)
            assert(len(self.turnTable)==len(self.evasion_turnTable))
        else:
            self.evasion_turnTable=self.turnTable
        self.dstAz_relative=getValueFromJsonKRD(self.modelConfig,"dstAz_relative",self.randomGen,False)

		# 上昇・下降に関する設定
        self.use_altitude_command=getValueFromJsonKRD(self.modelConfig,"use_altitude_command",self.randomGen,False)
        if(self.use_altitude_command):
            self.altTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"altTable",self.randomGen,
				[-8000.0,-4000.0,-2000.0,-1000.0,0.0,1000.0,2000.0,4000.0,8000.0])),dtype=np.float64)
            self.refAltInterval=getValueFromJsonKRD(self.modelConfig,"refAltInterval",self.randomGen,1000.0)
        else:
            self.pitchTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"pitchTable",self.randomGen,
				[-45.0,-20.0,-10.0,-5.0,0.0,5.0,10.0,20.0,45.0])),dtype=np.float64)
            self.pitchTable*=deg2rad(1.0)
            self.refAltInterval=1.0
        
        # 加減速に関する設定
        self.accelTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"accelTable",self.randomGen,[-2.0,0.0,2.0])),dtype=np.float64)
        self.always_maxAB=getValueFromJsonKRD(self.modelConfig,"always_maxAB",self.randomGen,False)
        
        # 射撃に関する設定
        self.use_Rmax_fire=getValueFromJsonKRD(self.modelConfig,"use_Rmax_fire",self.randomGen,False)
        if(self.use_Rmax_fire):
            self.shotIntervalTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"shotIntervalTable",self.randomGen,
				[5.0,10.0,20.0,40.0,80.0])),dtype=np.float64)
            self.shotThresholdTable=np.array(sorted(getValueFromJsonKRD(self.modelConfig,"shotThresholdTable",self.randomGen,
				[0.0,0.25,0.5,0.75,1.0])),dtype=np.float64)
        #行動制限に関する設定
        #  高度制限に関する設定
        self.altMin=getValueFromJsonKRD(self.modelConfig,"altMin",self.randomGen,2000.0)
        self.altMax=getValueFromJsonKRD(self.modelConfig,"altMax",self.randomGen,15000.0)
        self.altitudeKeeper=AltitudeKeeper(modelConfig().get("altitudeKeeper",{}))
        # 同時射撃数の制限に関する設定
        self.maxSimulShot=getValueFromJsonKRD(self.modelConfig,"maxSimulShot",self.randomGen,4)
        # 下限速度の制限に関する設定
        self.minimumV=getValueFromJsonKRD(self.modelConfig,"minimumV",self.randomGen,150.0)
        self.minimumRecoveryV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryV",self.randomGen,180.0)
        self.minimumRecoveryDstV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryDstV",self.randomGen,200.0)
        nvec=[]
        for pIdx,parent in enumerate(self.parents.values()):
            nvec.append(len(self.turnTable))
            nvec.append(len(self.altTable) if self.use_altitude_command else len(self.pitchTable))
            if(not self.always_maxAB):
                nvec.append(len(self.accelTable))
            nvec.append(self.maxTrackNum["Enemy"]+1)
            if(self.use_Rmax_fire):
                if(len(self.shotIntervalTable)>1):
                    nvec.append(len(self.shotIntervalTable))
                if(len(self.shotThresholdTable)>1):
                    nvec.append(len(self.shotThresholdTable))
        self.totalActionDim=1
        self.actionDims=np.zeros([len(nvec)], dtype=int)
        for i in range(len(nvec)):
            self.totalActionDim*=nvec[i]
            self.actionDims[i]=nvec[i]

        self.actionInfos=[self.ActionInfo() for _ in self.parents]


    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(78,))


    def action_space(self):
        return spaces.MultiDiscrete(self.actionDims)


    def validate(self):
        #Rulerに関する情報の取得
        rulerObs=self.manager.getRuler()().observables()
        self.dOut=rulerObs["dOut"] # 戦域中心から場外ラインまでの距離
        self.dLine=rulerObs["dLine"] # 戦域中心から防衛ラインまでの距離
        self.teamOrigin=self.TeamOrigin(self.own==rulerObs["eastSider"],self.dLine) # 陣営座標系変換クラス定義


    def makeObs(self):
        # 味方機(自機含む)
        self.ourMotion=[]
        self.ourObservables=[]

        for pIdx,parent in enumerate(self.parents.values()):
            if(parent.isAlive()):
                firstAlive=parent
                break
        for pIdx,parent in enumerate(self.parents.values()):
            if(parent.isAlive()):
                #残存していればobservablesそのもの
                self.ourMotion.append(parent.observables["motion"]())
                print(self.normalizer.normalize("Observation",str(parent.observables)))
                self.ourObservables.append(parent.observables)
                myMotion=MotionState(parent.observables["motion"])

            else:
                self.ourMotion.append({})
                #被撃墜or墜落済なら本体の更新は止まっているので残存している親が代理更新したものを取得(誘導弾情報のため)
                self.ourObservables.append(
                    firstAlive.observables.at_p("/shared/fighter").at(parent.getFullName()))

        # 彼機(味方の誰かが探知しているもののみ)
        # 観測されている航跡を、自陣営の機体に近いものから順にソートしてlastTrackInfoに格納する。
        # lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
        def distance(track):
            ret=-1.0
            for pIdx,parent in enumerate(self.parents.values()):
                if(parent.isAlive()):
                    myMotion=MotionState(parent.observables["motion"])
                    tmp=np.linalg.norm(track.posI()-myMotion.pos)
                    if(ret<0 or tmp<ret):
                        ret=tmp
            return ret
        for pIdx,parent in enumerate(self.parents.values()):
            if(parent.isAlive()):
                self.lastTrackInfo=sorted([Track3D(t) for t in parent.observables.at_p("/sensor/track")],key=distance) # type: ignore
                break

        # 味方誘導弾(射撃時刻が古いものから最大N発分)
        # 味方の誘導弾を射撃時刻の古い順にソート
        def launchedT(m):
            return m["launchedT"]() if m["isAlive"]() and m["hasLaunched"]() else np.inf
        self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)

        f_vec = [0.0]*6
        for fIdx in range(len(self.ourMotion)):
            if(fIdx>=self.maxTrackNum["Friend"]):
                break

            if(self.ourObservables[fIdx]["isAlive"]()):
                #初期弾数
                numMsls=self.ourObservables[fIdx].at_p("/spec/weapon/numMsls")()
                #残弾数
                remMsls=self.ourObservables[fIdx].at_p("/weapon/remMsls")()

                f_vec[3*fIdx] = remMsls/numMsls
                #MWS検出情報
                def angle(track):
                    return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
                mws=sorted([Track2D(t) for t in self.ourObservables[fIdx].at_p("/sensor/mws/track")],key=angle)
                if len(mws):
                    f_vec[3*fIdx+1] = mws[0].dirI()[0]
                    f_vec[3*fIdx+2] = mws[0].dirI()[1]

        # observationの作成
        om_vec = []
        for om in self.ourMotion:
            if len(om)==0:
                om_vec += [0.0]*6
            else:
                om_vec += om['pos']+om['vel']

        lt_vec = []
        for lt in self.lastTrackInfo:
            lt_vec += list(lt.pos)+list(lt.vel)
        n_lt = len(lt_vec)
        if n_lt < 12:
            lt_vec += [0.0]*(12-n_lt)

        msls = [m() for m in self.msls]
        m_vec = []
        for msl in msls:
            if msl['isAlive']:
                m_vec += msl['motion']['pos']+msl['motion']['vel']
            else:
                m_vec += [0.0]*6

        vec = om_vec + lt_vec + m_vec + f_vec

        return np.array(vec, dtype=np.float32)


    def deploy(self,action):
        print("action:",action)
        self.last_action_obs=np.zeros([self.maxTrackNum["Friend"],self.last_action_dim],dtype=np.float32)
        action_idx=0

        for pIdx,parent in enumerate(self.parents.values()):
            if(not parent.isAlive()):
                continue
            actionInfo=self.actionInfos[pIdx]
            myMotion=MotionState(parent.observables["motion"])
			#左右旋回
            deltaAz=self.turnTable[action[action_idx]]
            def angle(track):
                return -np.dot(track.dirI(),myMotion.relBtoP(np.array([1,0,0])))
            mws=sorted([Track2D(t) for t in parent.observables.at_p("/sensor/mws/track")],key=angle)
            if(len(mws)>0 and self.use_override_evasion):
                deltaAz=self.evasion_turnTable[action[action_idx]]
                dr=np.zeros([3])
                for m in mws:
                    dr+=m.dirI()
                dr/=np.linalg.norm(dr)
                dstAz=atan2(-dr[1],-dr[0])+deltaAz
                actionInfo.dstDir=np.array([cos(dstAz),sin(dstAz),0])
            elif(self.dstAz_relative):
                actionInfo.dstDir=myMotion.relHtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
            else:
                actionInfo.dstDir=self.teamOrigin.relBtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
            action_idx+=1
            dstAz=atan2(actionInfo.dstDir[1],actionInfo.dstDir[0])
            self.last_action_obs[pIdx,0]=dstAz

			#上昇・下降
            if(self.use_altitude_command):
                refAlt=round(-myMotion.pos(2)/self.refAltInterval)*self.refAltInterval
                actionInfo.dstAlt=max(self.altMin,min(self.altMax,refAlt+self.altTable[action[action_idx]]))
                dstPitch=0#dstAltをcommandsに与えればSixDoFFighter::FlightControllerのaltitudeKeeperで別途計算されるので0でよい。
            else:
                dstPitch=self.pitchTable[action[action_idx]]
            action_idx+=1
            actionInfo.dstDir=np.array([actionInfo.dstDir[0]*cos(dstPitch),actionInfo.dstDir[1]*cos(dstPitch),-sin(dstPitch)])
            self.last_action_obs[pIdx,1]=actionInfo.dstAlt if self.use_altitude_command else dstPitch

			#加減速
            V=np.linalg.norm(myMotion.vel)
            if(self.always_maxAB):
                actionInfo.asThrottle=True
                actionInfo.keepVel=False
                actionInfo.dstThrottle=1.0
                self.last_action_obs[pIdx,2]=1.0
            else:
                actionInfo.asThrottle=False
                accel=self.accelTable[action[action_idx]]
                action_idx+=1
                actionInfo.dstV=V+accel # type: ignore
                actionInfo.keepVel = accel==0.0
                self.last_action_obs[pIdx,2]=accel/max(self.accelTable[-1],self.accelTable[0])
            #下限速度の制限
            if(V<self.minimumV):
                actionInfo.velRecovery=True
            if(V>=self.minimumRecoveryV):
                actionInfo.velRecovery=False
            if(actionInfo.velRecovery):
                actionInfo.dstV=self.minimumRecoveryDstV
                actionInfo.asThrottle=False

            #射撃
            #actionのパース
            shotTarget=action[action_idx]-1
            action_idx+=1
            if(self.use_Rmax_fire):
                if(len(self.shotIntervalTable)>1):
                    shotInterval=self.shotIntervalTable[action[action_idx]]
                    action_idx+=1
                else:
                    shotInterval=self.shotIntervalTable[0]
                if(len(self.shotThresholdTable)>1):
                    shotThreshold=self.shotThresholdTable[action[action_idx]]
                    action_idx+=1
                else:
                    shotThreshold=self.shotThresholdTable[0]
            #射撃可否の判断、射撃コマンドの生成
            flyingMsls=0
            for msl in parent.observables.at_p("/weapon/missiles"):
                if(msl.at("isAlive")() and msl.at("hasLaunched")()):
                    flyingMsls+=1
            if(
                 shotTarget>=0 and
                 shotTarget<len(self.lastTrackInfo) and
                 parent.isLaunchableAt(self.lastTrackInfo[shotTarget]) and
                 flyingMsls<self.maxSimulShot
            ):
                 if(self.use_Rmax_fire):
                    rMin=np.inf
                    t=self.lastTrackInfo[shotTarget]
                    r=self.calcRNorm(parent,myMotion,t)
                    if(r<=shotThreshold):
                        #射程の条件を満たしている
                        if(not t.truth in actionInfo.lastShotTimes):
                            actionInfo.lastShotTimes[t.truth]=0.0
                        if(self.manager.getTime()-actionInfo.lastShotTimes[t.truth]>=shotInterval):
                            #射撃間隔の条件を満たしている
                            actionInfo.lastShotTimes[t.truth]=self.manager.getTime()
                        else:
                            #射撃間隔の条件を満たさない
                            shotTarget=-1
                    else:
                        #射程の条件を満たさない
                        shotTarget=-1
            else:
                shotTarget=-1
            self.last_action_obs[pIdx,3+(shotTarget+1)]=1
            if(shotTarget>=0):
                actionInfo.launchFlag=True
                actionInfo.target=self.lastTrackInfo[shotTarget]
            else:
                actionInfo.launchFlag=False
                actionInfo.target=Track3D()
            
            self.observables[parent.getFullName()]["decision"]={
                "Roll":("Don't care"),
                "Fire":(actionInfo.launchFlag,actionInfo.target.to_json())
            }
            if(len(mws)>0 and self.use_override_evasion):
                self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
            else:
                if(self.dstAz_relative):
                    self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_BODY",deltaAz)
                else:
                    self.observables[parent.getFullName()]["decision"]["Horizontal"]=("Az_NED",dstAz)
            if(self.use_altitude_command):
                self.observables[parent.getFullName()]["decision"]["Vertical"]=("Pos",-actionInfo.dstAlt)
            else:
                self.observables[parent.getFullName()]["decision"]["Vertical"]=("El",-dstPitch)
            if(actionInfo.asThrottle):
                self.observables[parent.getFullName()]["decision"]["Throttle"]=("Throttle",actionInfo.dstThrottle)
            else:
                self.observables[parent.getFullName()]["decision"]["Throttle"]=("Vel",actionInfo.dstV)


    def control(self):
		#Setup collision avoider
        avoider=StaticCollisionAvoider2D()
        #北側
        c={
            "p1":np.array([+self.dOut,-5*self.dLine,0]),
            "p2":np.array([+self.dOut,+5*self.dLine,0]),
            "infinite_p1":True,
            "infinite_p2":True,
            "isOneSide":True,
            "inner":np.array([0.0,0.0]),
            "limit":self.dOutLimit,
            "threshold":self.dOutLimitThreshold,
            "adjustStrength":self.dOutLimitStrength,
        }
        avoider.borders.append(LinearSegment(c))
        #南側
        c={
            "p1":np.array([-self.dOut,-5*self.dLine,0]),
            "p2":np.array([-self.dOut,+5*self.dLine,0]),
            "infinite_p1":True,
            "infinite_p2":True,
            "isOneSide":True,
            "inner":np.array([0.0,0.0]),
            "limit":self.dOutLimit,
            "threshold":self.dOutLimitThreshold,
            "adjustStrength":self.dOutLimitStrength,
        }
        avoider.borders.append(LinearSegment(c))
        #東側
        c={
            "p1":np.array([-5*self.dOut,+self.dLine,0]),
            "p2":np.array([+5*self.dOut,+self.dLine,0]),
            "infinite_p1":True,
            "infinite_p2":True,
            "isOneSide":True,
            "inner":np.array([0.0,0.0]),
            "limit":self.dOutLimit,
            "threshold":self.dOutLimitThreshold,
            "adjustStrength":self.dOutLimitStrength,
        }
        avoider.borders.append(LinearSegment(c))
        #西側
        c={
            "p1":np.array([-5*self.dOut,-self.dLine,0]),
            "p2":np.array([+5*self.dOut,-self.dLine,0]),
            "infinite_p1":True,
            "infinite_p2":True,
            "isOneSide":True,
            "inner":np.array([0.0,0.0]),
            "limit":self.dOutLimit,
            "threshold":self.dOutLimitThreshold,
            "adjustStrength":self.dOutLimitStrength,
        }
        avoider.borders.append(LinearSegment(c))
        for pIdx,parent in enumerate(self.parents.values()):
            if(not parent.isAlive()):
                continue
            actionInfo=self.actionInfos[pIdx]
            myMotion=MotionState(parent.observables["motion"])
            pos=myMotion.pos
            vel=myMotion.vel
            #戦域逸脱を避けるための方位補正
            actionInfo.dstDir=avoider(myMotion,actionInfo.dstDir)
            #高度方向の補正(actionがピッチ指定の場合)
            if(not self.use_altitude_command):
                n=sqrt(actionInfo.dstDir[0]*actionInfo.dstDir[0]+actionInfo.dstDir[1]*actionInfo.dstDir[1])
                dstPitch=atan2(-actionInfo.dstDir[2],n)
                #高度下限側
                bottom=self.altitudeKeeper(myMotion,actionInfo.dstDir,self.altMin)
                minPitch=atan2(-bottom[2],sqrt(bottom[0]*bottom[0]+bottom[1]*bottom[1]))
                #高度上限側
                top=self.altitudeKeeper(myMotion,actionInfo.dstDir,self.altMax)
                maxPitch=atan2(-top[2],sqrt(top[0]*top[0]+top[1]*top[1]))
                dstPitch=max(minPitch,min(maxPitch,dstPitch))
                cs=cos(dstPitch)
                sn=sin(dstPitch)
                actionInfo.dstDir=np.array([actionInfo.dstDir[0]/n*cs,actionInfo.dstDir[1]/n*cs,-sn])
            self.commands[parent.getFullName()]={
                "motion":{
                    "dstDir":actionInfo.dstDir
                },
                "weapon":{
                    "launch":actionInfo.launchFlag,
                    "target":actionInfo.target.to_json()
                }
            }
            if(self.use_altitude_command):
                self.commands[parent.getFullName()]["motion"]["dstAlt"]=actionInfo.dstAlt
            if(actionInfo.asThrottle):
                self.commands[parent.getFullName()]["motion"]["dstThrottle"]=actionInfo.dstThrottle
            elif(actionInfo.keepVel):
                self.commands[parent.getFullName()]["motion"]["dstAccel"]=0.0
            else:
                self.commands[parent.getFullName()]["motion"]["dstV"]=actionInfo.dstV
            actionInfo.launchFlag=False

    
    def calcRHead(self,parent,myMotion,track):
        #相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程(RHead)を返す。
        rt=track.posI()
        vt=track.velI()
        rs=myMotion.pos
        vs=myMotion.vel
        return parent.getRmax(rs,vs,rt,vt,np.pi)
    
    
    def calcRTail(self,parent,myMotion,track):
        #相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程(RTail)を返す。
        rt=track.posI()
        vt=track.velI()
        rs=myMotion.pos
        vs=myMotion.vel
        return parent.getRmax(rs,vs,rt,vt,0.0)

    
    def calcRNorm(self,parent,myMotion,track):
        #RTail→0、RHead→1として正規化した距離を返す。
        RHead=self.calcRHead(parent,myMotion,track)
        RTail=self.calcRTail(parent,myMotion,track)
        rs=myMotion.pos
        rt=track.posI()
        r=np.linalg.norm(rs-rt)-RTail
        delta=RHead-RTail
        outRangeScale=100000.0
        if(delta==0):
            if(r<0):
                r=r/outRangeScale
            elif(r>0):
                r=1+r/outRangeScale
            else:
                r=0
        else:
            if(r<0):
                r=r/outRangeScale
            elif(r>delta):
                r=1+(r-delta)/outRangeScale
            else:
                r/=delta
        return r
