import numpy as np
import matplotlib.pyplot as plt

import mnist_loader
from network import Network


# Загружаем сеть
net = Network.load("models/784-30-10.pkl")

# Загружаем данные
_, _, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)

current = 0


fig, (ax_img, ax_bar) = plt.subplots(
    1,
    2,
    figsize=(10, 5)
)


def draw(index):
    ax_img.clear()
    ax_bar.clear()

    x, y = test_data[index]

    output = net.feedforward(x)
    prediction = np.argmax(output)

    # Картинка
    ax_img.imshow(
        x.reshape(28, 28),
        cmap="gray"
    )

    ax_img.set_title(
        f"True: {y}\nPredicted: {prediction}"
    )

    ax_img.axis("off")

    # Активации выходного слоя
    ax_bar.bar(
        range(10),
        output.flatten()
    )

    ax_bar.set_xticks(range(10))
    ax_bar.set_xlabel("Digit")
    ax_bar.set_ylabel("Activation")
    ax_bar.set_title("Output Layer")

    fig.suptitle(
        f"Image {index + 1}/{len(test_data)}"
    )

    fig.canvas.draw_idle()


def on_key(event):
    global current

    if event.key == "right":
        current = (current + 1) % len(test_data)

    elif event.key == "left":
        current = (current - 1) % len(test_data)

    else:
        return

    draw(current)


fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)

draw(current)

plt.tight_layout()
plt.show()