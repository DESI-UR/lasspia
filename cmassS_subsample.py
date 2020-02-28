from cmassS_coarse import cmassS_coarse

class cmassS_subsample(cmassS_coarse):
    def binningRA(self): return {"bins": 220, "range":(0,10)}
    def binningDec(self): return {"bins":156, "range":(0,6)}
