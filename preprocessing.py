from pathlib import Path
import yaml
from datetime import datetime


def normalise_text(text, lowercase=False):
    """
    Basic text normalisation.
    Additional cleaning steps can be added here later.
    """
    
    lines = text.splitlines()
    text = "\n".join(lines[6:])
    
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def create_experiment_folder(
    experiment_name,
    source_folder="source_files",
    experiments_folder="experiments",
    lowercase=False, 
    granularity = "section",
    prompt = "No prompt properly specified. Please edit the create_experiment_folder function call to include a prompt string."
):
    """
    Creates experiment structure and preprocesses source files.
    """

    experiment_path = Path(experiments_folder) / experiment_name

    input_path = experiment_path / "input_files"
    output_path = experiment_path / "output_files"
    logs_path = experiment_path / "logs"
    prompt_path = experiment_path / "prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    # Create folders
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(exist_ok=True)
    logs_path.mkdir(exist_ok=True)

    # Process source files
    source_path = Path(source_folder)

    processed_files = []

    combined_file = input_path / f"{experiment_name}_combined.txt"

    for file in source_path.glob("*.txt"):

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        processed_text = normalise_text(
            text,
            lowercase=lowercase
        )
        if granularity == "section":
            lines = processed_text.splitlines()
            chunk_size = 10
            sections = [
                "\n".join(lines[i:i + chunk_size])
                for i in range(0, len(lines), chunk_size)
            ]

            for i, section in enumerate(sections):
                section_file = input_path / f"{file.stem}_section_{i+1}.txt"
                with open(section_file, "w", encoding="utf-8") as f:
                    f.write(section)
                processed_files.append(section_file.name)
                
        elif granularity == "corpus":
           with open(combined_file, "a", encoding="utf-8") as g:
                g.write(text+"\n\n")
            
        else:
            output_file = input_path / file.name

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(processed_text)

        processed_files.append(file.name)


    # Create metadata
    metadata = {
        "experiment_name": experiment_name,
        "created": datetime.now().isoformat(),

        "source": {
            "folder": str(source_folder),
            "files_processed": processed_files
        },

        "preprocessing": {
            "lowercase": lowercase,
            "strip_whitespace": True
        }
    }


    with open(
        experiment_path / "metadata.yaml",
        "w",
        encoding="utf-8"
    ) as f:
        yaml.dump(
            metadata,
            f,
            sort_keys=False
        )


    return experiment_path



if __name__ == "__main__":

    experiment = create_experiment_folder(
        experiment_name="experiment_001_section_level",
        lowercase=False
    )

    print(f"Created: {experiment}")