from collections import Counter
import re


file_path = "./The Bitter Lesson.txt"

with open(file_path, "r") as file:
    content = file.read()

lines = content.split("\n")
line_count = len(lines)

words = re.findall(r"\b[a-z]+\b", content.lower())
word_count = len(words)

word_frequency = Counter(words)


print(f"行数: {line_count}")
print(f"总单词数: {word_count}")
print(f"单词出现频率 (按频率从高到低):")

for word, freq in word_frequency.most_common(10):
    print(f"{word} 出现 {freq} 次")

print(f"不同单词总数: {len(word_frequency)}")
