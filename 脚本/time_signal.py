"""
@file     time_signal.py
@author   YellowOrz
@detail   整点报时、间隔提醒
@note     pypi依赖pyttsx3
@data     2025/01/06
@example  python time_signal.py
@note     windows开机自启：任务计划程序=>创建任务，
            触发器：“开始任务”选择“登录时”，
            操作：程序选择python.exe，参数选择本脚本
            条件：取消“使用交流电才启动”
            设置：取消“如果任务运行时间超过一下...”
"""
import time
import pyttsx3

# 整点开始间隔提醒时间，单位为分钟，取值范围1-59，超出范围关闭间隔提醒
interval_min = 20

last_signal = -1

if __name__ == "__main__":
    if interval_min > 0 and interval_min < 60:
        print("[INFO] 开始整点报时、间隔提醒")
    else:
        interval_min = -1
        print("[INFO] 开始整点报时")
    # 初始化pyttsx3引擎
    engine = pyttsx3.init()
    
    while True:
        # 获取当前时间
        current_time = time.localtime()
        hour = current_time.tm_hour
        minute = current_time.tm_min
        
        # 如果是整点，进行语音报时
        time_str = ""
        if minute == 0:
            time_str = f"现在是{hour}点整"
            print(f"[INFO] {time_str}")
            last_signal = 0
        # 间隔提醒
        elif interval_min > 0 : 
            if minute < last_signal:
                m = minute - last_signal + 60
            else:
                m = minute - last_signal
            if m > interval_min:
                time_str = f"现在是{hour}点{minute}分"
                print(f"[INFO] {time_str}")
                last_signal = minute
        
        # 使用pyttsx3进行语音播报
        if time_str != "":
            engine.say(time_str)
            engine.runAndWait()
        
        # 每分钟检查一次
        time.sleep(60)