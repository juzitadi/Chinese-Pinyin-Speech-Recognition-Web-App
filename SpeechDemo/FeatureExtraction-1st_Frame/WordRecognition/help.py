import matplotlib.pyplot as plt
import numpy as np

def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def plot_curve(history):
    plt.style.use('ggplot')
    # Create sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Summarize history for accuracy
    axs[0].plot(range(1, len(history.history['accuracy']) + 1),history.history['accuracy'],'b')
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(history.history['accuracy']) + 1), len(history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'],'b')
    axs[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(history.history['loss']) + 1), len(history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')

    # Show the plot
    plt.show()


def smooth_curve(points, factor=0.8):
    '''
    :param points: acc或者loss的点
    :param factor: 
    :return: 
    '''
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_curve_with_smooth(history):
    plt.style.use('ggplot')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Summarize history for accuracy
    axs[0].plot(range(1, len(history.history['accuracy']) + 1), smooth_curve(acc),'b')
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), smooth_curve(val_acc))
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(history.history['accuracy']) + 1), len(history.history['accuracy']) / 10)
    axs[0].legend(['Smoothed training acc', 'Smoothed validation acc'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1, len(history.history['loss']) + 1), smooth_curve(loss),'b')
    axs[1].plot(range(1, len(history.history['val_loss']) + 1), smooth_curve(val_loss))
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(history.history['loss']) + 1), len(history.history['loss']) / 10)
    axs[1].legend(['Smoothed training loss', 'Smoothed validation loss'], loc='best')
    # Show the plot
    plt.show()


def predict_classes(model, images_test, labels_test):
    # Predict class of image using model
    class_pred = model.predict(images_test, batch_size=32)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred, axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == labels_test)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred


def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
    incorrect = (correct == False)

    # Images of the test-set that have been incorrectly classified.
    images_error = images_test[incorrect]

    # Get predicted classes for those images
    labels_error = labels_pred[incorrect]

    # Get true classes for those images
    labels_true = labels_test[incorrect]

    # Plot the first 9 images.
    plot_images(images=images_error[0:9],
                labels_true=labels_true[0:9],
                class_names=class_names,
                labels_pred=labels_error[0:9])


def plot_images(images, labels_true, class_names, labels_pred=None):
    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')

            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: " + labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()