import torch
import numpy as np
import matplotlib.pyplot as plt
from beat_this.inference import load_model


def plot_gradients_over_time(
        checkpoint_path = str,
        input_tensor = torch.Tensor,
        chosen_frame = int,
        custom_model_name = str,
        target = str,
):
    '''
    Shows the gradients of the chosen frame prediction (beats) w.r.t. the input tensor.

    Args:
        checkpoint_path: relative path to checkpoint file to evaluate
        input_tensor: shape (1, sequence length, 128), requires_grad=True
        chosen_frame: index of the frame to evaluate
        custom_model_name: custom name for the loaded model to show on the plot
        target: either "model_output" or "frontend_output"
    '''

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    model.eval()

    if target == "Model Output":
        # forward pass
        output = model(input_tensor)
        beat_output = output["beat"]

        # grad of the frame prediction w.r.t. input
        chosen_prediction = beat_output[0, chosen_frame].sum()
        chosen_prediction.backward()
    elif target == "Frontend Output":
        x_feat = model.frontend(input_tensor)
        x_feat.retain_grad()

        feat_scalar = x_feat[0, chosen_frame].sum()
        feat_scalar.backward()
    else:
        raise ValueError("target must be either 'Model Output' or 'Frontend Output'")

    gradients = input_tensor.grad

    # sum grads across the frequency axis
    summed_gradients = gradients[0].abs().sum(dim=1).cpu().numpy()

    # find the first idx where the gradient is above 0
    first_positive_index = np.argmax(summed_gradients > 0)
    #first_positive_value = summed_gradients[first_positive_index]

    # last idx where the gradient is above 0
    last_positive_index = summed_gradients.shape[0] - 1 - np.argmax((summed_gradients[::-1] > 0))

    # plot the summed gradients
    plt.figure(figsize=(4.5, 2.2))
    plt.plot(summed_gradients, label="Grads")
    plt.axvline(x=first_positive_index, color='red', linestyle='--', label=f"First > 0: {first_positive_index}")
    #plt.text(first_positive_index, 1, f"{first_positive_value:.12f}", color='red', fontsize=10, ha='right')
    plt.axvline(last_positive_index, color='green', linestyle='--', label=f"Last > 0: {last_positive_index}")
    plt.xlabel("Frames")
    plt.ylabel("Gradient Magnitude")
    if custom_model_name:
        #plt.title(f"{custom_model_name} | Frame {chosen_frame}")
        pass
    else:
        plt.title("Summed Gradients Over Time (Abs. Value)")
    plt.legend(loc="upper right")#, bbox_to_anchor=(0.96, 0.98))
    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.20, top=0.97)
    plt.ylim(0, 1)
    plt.yticks([0, 1])
    plt.savefig(f"plots/paper/grads_{custom_model_name}.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)
    #checkpoint_path = "small0"
    #checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=0 cc=1.ckpt"
    #checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1.ckpt"
    #checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1.ckpt"
    checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1 sw256.ckpt"

    input_tensor = torch.randn(1, 1500, 128, requires_grad=True, device='cuda')
    chosen_frame = 750

    #custom_model_name = "Beat This! small"
    #custom_model_name = "Full Attn. & Causal Conv."
    #custom_model_name = "Causal Attn. & Vanilla Conv."
    #custom_model_name = "Causal Attn. & Causal Conv."
    custom_model_name = "Sliding Window Attn."

    plot_gradients_over_time(checkpoint_path, input_tensor, chosen_frame, custom_model_name, target="Frontend Output")
