import matplotlib.pyplot as plt
import os

labels = ['False Positive\n(Báo động nhầm)', 'False Negative\n(Bỏ lọt lừa đảo)']
sizes = [64, 23]
colors = ['#ff9999','#66b3ff']
explode = (0.05, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Biểu đồ tỷ lệ phân bố các trường hợp dự đoán sai')

output_path = r'e:\Dai Hoc\Nam 3\Xu ly ngon ngu\MailSentry\reports\evaluation\error_pie_chart.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved pie chart to {output_path}")
