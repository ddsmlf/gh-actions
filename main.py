import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from modele import SimpleImageClassifier

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bienvenue dans votre application de classification d'images de chats et no_cats")

        self.welcome_label = tk.Label(self.master, text="Bienvenue dans votre application de classification d'images de chats et no_cats")
        self.welcome_label.pack(pady=20)
        self.classifier = SimpleImageClassifier()
        self.continue_button = tk.Button(self.master, text="Continuer", command=self.show_main_interface)
        self.continue_button.pack()

    def show_main_interface(self):
        # Détruire les widgets de la page d'accueil
        self.welcome_label.destroy()
        self.continue_button.destroy()

        # Créer les widgets de l'interface principale
        self.image_label = tk.Label(self.master, text="Aucune image sélectionnée")
        self.image_label.pack()

        self.choose_button = tk.Button(self.master, text="Choisir une image", command=self.choose_image)
        self.choose_button.pack(side=tk.LEFT, padx=10)

        self.classify_button = tk.Button(self.master, text="Classer l'image", command=self.classify_image)
        self.classify_button.pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack(pady=20)

    def choose_image(self):
        file_path = filedialog.askopenfilename(title="Choisir une image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.selected_image_path = file_path

    def classify_image(self):
        if hasattr(self, 'selected_image_path'):
            detection = self.classifier.inferer_image(self.selected_image_path, "modele_poids.h5")
            result = "Image classée comme " + detection  
            self.result_label.config(text=result)
        else:
            messagebox.showinfo("Info", "Veuillez choisir une image avant de classer.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Classifier App")
    root.geometry("600x400")  # Taille de la fenêtre
    app = ImageClassifierApp(root)
    root.mainloop()