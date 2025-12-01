import dataclasses
from typing import List, Optional, Union, Any
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import LlamaTokenizer

try:
    from RadFM.models.radfm.modeling_llama import LlamaForCausalLM
except ImportError:
    print("Error: Could not import RadFM")


@dataclasses.dataclass
class MedicalCase:
    case_id: str
    image_path: str
    report_text: str


@dataclasses.dataclass
class VLMInput:
    content: List[Union[str, dict]]


class RadFMWrapper:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        print(f"Loading RadFM from {model_path}...")

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

        self.transforms = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def process_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
            return self.transforms(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros((3, 512, 512))

    def generate(self, prompt_payload: List[Union[str, dict]]) -> str:
        full_text_prompt = ""
        images_list = []

        for item in prompt_payload:
            if item["type"] == "text":
                full_text_prompt += item["text"]
            elif item["type"] == "image_url":
                full_text_prompt += " <image> "
                img_path = item["image_url"]["url"]
                images_list.append(self.process_image(img_path))

        if images_list:
            pixel_values = torch.stack(images_list).to(self.device).to(torch.float16)
            pixel_values = pixel_values.unsqueeze(0)
        else:
            pixel_values = None

        if not full_text_prompt.strip().endswith(":"):
            full_text_prompt += "\nAnswer:"

        input_ids = self.tokenizer(full_text_prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=pixel_values,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()


class VLMRefiner:
    def __init__(self, vlm: Any, k_neighbors: int = 5):
        self.vlm = vlm
        self.k_neighbors = k_neighbors

        self.persona = "You are an expert radiology assistant for CT interpretation and report generation."
        self.task = (
            "Refine the provided `base report` into a `final report` using few-shot exemplars "
            "to match expert style, structure, and clinical detail."
        )
        self.context_desc = (
            "1. Input: [input CT scan, `base report`]\n"
            "2. Exemplars: Top-k visually similar CT scans with reports, retrieved from Stage 1."
        )
        self.procedure = (
            "1. Analyze the input CT scan and its `base report`.\n"
            "2. Review exemplar CT-report pairs to learn reporting style and diagnostic detail.\n"
            "3. Synthesize the `final report` in a consistent radiology style.\n"
            "4. Adapt exemplar guidance without copying or redundancy.\n"
            "5. Output only the refined report."
        )
        self.instructions = (
            "1. Do not include any explanations or meta-comments.\n"
            "2. Ensure the `final report` is clinically precise and aligns with clinical conventions."
        )

    def construct_prompt(
        self, target_case: MedicalCase, neighbors: List[MedicalCase]
    ) -> List[Union[str, dict]]:
        prompt_sequence = []

        system_text = (
            f"Persona: {self.persona}\n"
            f"Task: {self.task}\n"
            f"Context Provided:\n{self.context_desc}\n"
            f"Procedure:\n{self.procedure}\n"
            f"Instructions:\n{self.instructions}\n\n"
            "Here are the Exemplars (Context Item 2):"
        )
        prompt_sequence.append({"type": "text", "text": system_text})

        for idx, neighbor in enumerate(neighbors):
            prompt_sequence.append(
                {"type": "text", "text": f"\nExample Case {idx+1} Image:"}
            )
            prompt_sequence.append(
                {
                    "type": "image_url",
                    "image_url": {"url": neighbor.image_path},
                    "label": f"Example Case {idx+1}",
                }
            )
            prompt_sequence.append(
                {
                    "type": "text",
                    "text": f"\nExpert Report for Example {idx+1}:\n{neighbor.report_text}\n\n",
                }
            )

        prompt_sequence.append(
            {
                "type": "text",
                "text": "Now, here is the Input (Context Item 1) to refine. \nTarget Input Scan:",
            }
        )

        prompt_sequence.append(
            {
                "type": "image_url",
                "image_url": {"url": target_case.image_path},
                "label": "Target Input Scan",
            }
        )

        prompt_sequence.append(
            {
                "type": "text",
                "text": f"\nBase Report (Generated by Stage 1):\n{target_case.report_text}\n\n",
            }
        )

        prompt_sequence.append(
            {"type": "text", "text": "Generate the final refined report:"}
        )

        return prompt_sequence

    def refine_report(self, input_image_path: str, base_report_text: str) -> str:
        target_case = MedicalCase(
            case_id="target_001",
            image_path=input_image_path,
            report_text=base_report_text,
        )

        neighbors = self.retrieve_nearest_neighbors(input_embedding=None)

        prompt_payload = self.construct_prompt(target_case, neighbors)

        print("Sending prompt to RadFM...")
        final_report = self.vlm.generate(prompt_payload)

        return final_report


if __name__ == "__main__":
    RADFM_MODEL_PATH = "/path/to/RadFM/checkpoints"

    try:
        radfm_model = RadFMWrapper(model_path=RADFM_MODEL_PATH)

        refiner = VLMRefiner(vlm=radfm_model, k_neighbors=5)

        input_scan = "path/to/ct_scan_image"
        base_report = "report generated in stage 1"

        final_report = refiner.refine_report(input_scan, base_report)

        print("-" * 30)
        print(final_report)
        print("-" * 30)

    except Exception as e:
        print(f"Setup failed: {e}")
        print(
            "Ensure you have set RADFM_MODEL_PATH to the correct directory containing the model weights."
        )
