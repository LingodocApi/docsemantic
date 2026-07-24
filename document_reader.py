import os

class DocumentReader:
    def __init__(self, directory_path="./documents"):
        """
        Sets the directory where your text documents are kept.
        Creates the folder if it doesn't exist yet.
        """
        self.directory_path = directory_path
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

    def load_documents(self):
        """
        Scans the folder and reads the text out of every single .txt file.
        Returns a clean list of text strings.
        """
        documents = []
        
        # Check if the folder exists and loop through all files inside it
        if os.path.exists(self.directory_path):
            for filename in os.listdir(self.directory_path):
                # Only read files that end in .txt
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.directory_path, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        documents.append(file.read())
                        
        return documents