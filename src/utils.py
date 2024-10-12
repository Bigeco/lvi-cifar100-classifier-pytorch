import os
import json
from datetime import datetime


def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc

class Logger:
    def __init__(self, log_dir, headers, resume=False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "log.txt")
        self.headers = headers
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        if not resume:
            mode = "w"  # 새 학습 세션: 파일을 새로 만들거나 덮어쓴다.
        else:
            mode = "a"  # 학습 재개: 기존 파일에 추가한다.

        self.f = open(self.log_file, mode)

        if resume:
            self.f.write("\n\n--- Resuming Training ---\n\n")

        header_str = "\t".join(headers + ["EndTime."])
        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        log_str = self.print_str.format(*args, now_time)
        self.f.write(log_str + "\n")
        self.f.flush()
        print(log_str)

    def write_hp(self, hp):
        json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()
