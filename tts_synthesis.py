import pyttsx3
import os

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)

    def synthesize(self, text, output_path):
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return os.path.abspath(output_path)
        except Exception as e:
            print(f"TTS合成失败: {str(e)}")
            return None

if __name__ == "__main__":
    tts = TTSEngine()
    result_path = tts.synthesize("测试合成内容", "output.mp3")
    print(f"音频保存路径: {result_path}")