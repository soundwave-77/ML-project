import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import argparse


PROJECT_PATH = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_PATH / "outputs"
IMG_PATH = PROJECT_PATH / "data" / "train_images"

assert IMG_PATH.exists(), "Image path does not exist"
assert OUTPUT_PATH.exists(), "Output path does not exist"

def create_report(pred_path: Path):
    # join predictions with test data
    df = pd.read_csv(pred_path)

    # Obtain values with highest deal probability
    top_9 = df.nlargest(9, 'deal_probability')
    plot_grid(top_9, OUTPUT_PATH / "top_deal_probability.png")
    
    # Obtain values with lowest deal probability
    bottom_9 = df.nsmallest(9, 'deal_probability')
    plot_grid(bottom_9, OUTPUT_PATH / "bottom_deal_probability.png")

    # Save the report as .html file
    save_report(top_9, bottom_9, OUTPUT_PATH / "report.html")


def plot_grid(data: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (_, row) in zip(axes.flatten(), data.iterrows()):
        # Load the image
        img_path = IMG_PATH / f"{row['image']}.jpg" if row['image'] else None
        if Path(img_path).exists():
            img = plt.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', horizontalalignment='center', verticalalignment='center')
        # Set title with description and probability
        # title = f"{row.get('description', 'No Description')}\nProbability: {row['deal_probability']:.2f}"
        title = f"{row.get('title', 'No Title')}\nProbability: {row['deal_probability']:.2f}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_report(top_data: pd.DataFrame, bottom_data: pd.DataFrame, output_html: Path):
    # Ensure the templates directory exists
    templates_dir = OUTPUT_PATH
    templates_dir.mkdir(exist_ok=True)
    
    # Create a simple HTML template if it doesn't exist
    template_path = templates_dir / "report_template.html"
    if not template_path.exists():
        template_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deal Probability Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                h2 { text-align: center; }
                .section { margin-bottom: 50px; }
                .grid-image { width: 300px; height: 300px; }
            </style>
        </head>
        <body>
            <h1>Deal Probability Report</h1>
            
            <div class="section">
                <h2>Top 9 Deals</h2>
                <img src="top_deal_probability.png" alt="Top Deals" class="grid-image">
            </div>
            
            <div class="section">
                <h2>Bottom 9 Deals</h2>
                <img src="bottom_deal_probability.png" alt="Bottom Deals" class="grid-image">
            </div>
        </body>
        </html>
        """
        with open(template_path, 'w') as f:
            f.write(template_content)
    
    # Render the HTML report
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("report_template.html")
    html_content = template.render()
    
    with open(output_html, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="Create a report from predictions.")
    parser.add_argument('--pred-path', type=str, required=True, help='Path to the predictions CSV file.')
    args = parser.parse_args()

    create_report(args.pred_path)
