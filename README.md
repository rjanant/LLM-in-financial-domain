<h1>Large language model number handling in the Finance Domain</h1>

Abstract:
The integration of large language models (LLMs) in the financial domain requires scrupulous attention to numerical precision and the ability to process diverse data types. In this study, we enhance the capabilities of open-source Llama models through an innovative fine-tuning approach that leverages a hybrid dataset, combining tabular data, conversational histories, and contextual information. We employ specialized prompting, few-shot learning, and Chain of Thought reasoning to address complex numerical tasks inherent to financial analysis. Furthermore, we guide the models to generate Python scripts aimed at bolstering numerical accuracy. Our approach offers valuable insights into the potential and challenges of LLMs in complex numerical reasoning within financial contexts, illuminating areas for further enhancement. Our findings highlight the ongoing difficulties in merging varied data forms within LLM frameworks and underscore the necessity for continued refinement of these models to meet the rigorous demands of financial data analysis. This research not only advances our understanding of LLM applications in finance but also sets a foundational framework for future explorations aimed at enhancing the robustness and precision of financial data processing.


To ensure that you can reproduce our results accurately, please follow these detailed steps. This guide is designed to assist users of all levels, including those new to Python and virtual environments.

<h4>Prerequisites</h4>
Before you start, ensure you have Anaconda or Miniconda installed on your system to manage environments and dependencies. If you don't have Anaconda installed, please download it from Anaconda's official site.

<h4>Step-by-Step Guide</h4>
1. Create a Conda Environment
Create a new isolated environment to avoid conflicts between package versions. Open your terminal or Anaconda Prompt and run the following command:

<code> conda create --name myenv python=3.9</code> 

Replace myenv with a name of your choice for the new environment.

2. Activate the Environment and Install Dependencies
Activate the newly created environment:

<code> conda activate myenv </code>


Next, install all required packages using the requirements.txt file:

<code> pip install -r requirements.txt </code>

Ensure your requirements.txt file is in the same directory as your terminal path or provide the path to the file.

3. Run the Code in Jupyter Notebook
If you don’t have Jupyter Notebook installed in your environment, first install it using:

<code> pip install notebook </code>

Start the Jupyter Notebook server:

<code> jupyter notebook </code>

In the Jupyter interface, create a new notebook and manually copy the .py script code into separate cells in the notebook. Execute the cells sequentially by pressing Shift + Enter.

4. Reproduce Datasets
Use the scripts provided in the 'preprocess_raw_dataset' notebook to generate and merge datasets as required. Make sure to follow the notebook instructions for the correct order of operations.

5. Seed Values for Reproducibility
The code includes predefined seed values to ensure reproducibility. Make sure not to alter these unless necessary for your experiments, as they ensure that results are consistent across different runs.

6. Running Fine-Tuning Scripts
To run fine-tuning scripts with distributed training, use the following command:

<code> torchrun --nproc_per_node=8 your_file_name.py </code>
 
Replace your_file_name.py with the name of your script. This command utilizes 8 processes per node, designed for multi-GPU setups.

7. Hardware Requirements
The fine-tuning phase was conducted on a setup with 8 NVIDIA GeForce RTX 3090 GPUs. The code generation experiments were performed on a single NVIDIA A100 GPU available via Colab Pro. These specifications are recommended for optimal performance and exact reproducibility of the results.

If you encounter any issues:

Ensure all commands are typed correctly and executed in the newly created Conda environment.
Verify that your hardware setup meets the requirements for the fine-tuning and code generation experiments.
Check that the requirements.txt file includes all necessary packages with their correct versions.

**Note:** Although multiple evaluation metrics are defined in this repository, our paper primarily utilizes Average Permissive Accuracy and Average Exact Accuracy. These two metrics were selected for their relevance to the specific tasks discussed, including Relation-Extraction. Consequently, other metrics available here were not adopted for use in the paper.
## Citation
Please cite this software using the metadata from `CITATION.cff`. Here's an example citation in APA style:

Raj, Anant (2024).  Large language model number handling in the Finance Domain (Version 1.0.0) [Software]. Available at https://github.com/rjanant/disseration.git.
