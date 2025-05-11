import os
from data import srdata

class Demo(srdata.SRData):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        super(Demo, self).__init__(args, name=name, train=False, benchmark=False)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.dir_hr = os.path.join(self.apath, 'HR')  # 없으면 빈 폴더라도 생성하세요
        self.ext = ('.jpg', '.png')

    def _scan(self):
        names_lr = [os.path.join(self.dir_lr, '0001_Low.jpg')]  # 명확한 이미지 지정
        names_hr = [os.path.join(self.dir_hr, '0001.png')]  # HR은 옵션(없다면 임의 지정 가능)
        names_lr = [names_lr]  # 스케일 차원 대응
        return names_hr, names_lr
