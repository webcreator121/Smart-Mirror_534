## Installation and Setup

Before running the project, make sure that all the required Python libraries are installed on your computer. The project relies on external libraries to perform its operations, particularly:

- **OpenCV (`opencv-python`)** – Used for accessing the webcam, capturing video frames, and processing images.
- **DeepFace (`deepface`)** – Used for AI-based facial analysis and emotion recognition.

If these libraries are not installed, the program may encounter import errors and fail to run.

### Installing Dependencies

You can install the required libraries using Python's package manager, `pip`. Open a terminal or command prompt and execute:

```bash
python -m pip install opencv-python deepface
```

Wait for the installation to complete before proceeding. Make sure you have a working Python installation and that `pip` is properly configured.

### Running the Project

Once all the dependencies have been installed, follow these steps to launch the Smart Mirror application:

1. Navigate to the project folder containing the `mirror2.py` file.
2. Open a terminal or command prompt in that folder.
3. Execute the following command:

   ```bash
   python mirror2.py
   ```

4. Press **Enter** and wait for the application to initialize.
5. Once initialized, the webcam window should appear, allowing the Smart Mirror application to begin processing video input.

### Important Notes

- Make sure your webcam is connected and accessible to your computer.
- Ensure that Python is added to your system's PATH so that the `python` command works in the terminal.
- The first execution may take longer because DeepFace might need to download or initialize its required AI models.
- Allow the application to access your webcam if your operating system requests permission.
- If the program fails to launch, check whether all dependencies were installed successfully and whether you are running the command from the correct project directory.

**You're all set!** Once the application starts successfully, the Smart Mirror is ready to use.
