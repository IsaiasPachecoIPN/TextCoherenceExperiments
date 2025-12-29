
from lime.lime_text import LimeTextExplainer
from IPython.display import HTML, display
import re
import logging
import coloredlogs
import shap
import torch
import numpy            as np
import scipy            as sp
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt="%(levelname)s[Line:%(lineno)d]:%(message)s")

class ModelExplainer:

    def __init__(self,):
        self.model              = None
        self.tokenizer          = None
        self.class_names        = None
        self.model_type         = None
        self.model_name         = None
        self.dataset            = None
        self.bow                = True
        self.background_data    = None

    def preprocess_text(self, target_label, lower=False):
        """
        Preprocess the target_label using the flag
        """
        if lower:
            self.dataset[target_label] = self.dataset[target_label].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # Remove specific special characters, including JSON problematic characters (like ", \, [], <>, {}, etc.)
        self.dataset[target_label] = self.dataset[target_label].apply(
            lambda x: re.sub(r'[\[\]<>\"\\\{\}]', '', x) if isinstance(x, str) else x
        )

        logger.info("Preprocessing finished")

    def _save_html_as_image(self, html_code, output_image_path, driver_path=None):
        """
        Helper method to save HTML code as an image using Selenium.
        """
        if not output_image_path:
            return

        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_code)
            temp_html_path = f.name

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--window-size=1200,800")
            
            # Try to find chromedriver if not provided
            if not driver_path:
                # Check if chromedriver is in PATH or use a default location if known, 
                # otherwise rely on user having it in PATH or installed via webdriver-manager
                pass 

            service = Service(driver_path) if driver_path else Service()
            driver = webdriver.Chrome(service=service, options=chrome_options)

            # Load the local HTML file
            file_url = Path(temp_html_path).as_uri()
            driver.get(file_url)
            
            # Give it a moment to render
            import time
            time.sleep(1)

            # Save screenshot
            driver.save_screenshot(f"{output_image_path}.png")
            driver.quit()
            
            if os.path.exists(f"{output_image_path}.png"):
                logger.info(f"Saved explanation image to {output_image_path}.png")
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            logger.warning("Ensure 'chromedriver' is installed and in your PATH, or provide 'driver_path'.")
        finally:
            # Clean up temp file
            if os.path.exists(temp_html_path):
                os.remove(temp_html_path)

    def get_lime_explanation(self, sample_idx, num_features, output_image_path=None, verbose=False, driver_path=None):
        """
        LIME explainer is used to explain the model's prediction on a given sample.

        Args:
            sample_idx (int): Index of the sample in the dataset to explain.
            num_features (int): Number of features (words) to show in the explanation.
            verbose (bool): If True, print additional information.

        Returns:
            explanation: LIME explanation object.
        """
        # Get the sample and target
        sample = self.dataset.iloc[sample_idx]['text']
        target = self.dataset.iloc[sample_idx]['score']

        if verbose:
            print('Sample:', sample)
            print('Target:', target)

        # Define the predictor function based on the model type
        if self.model_type == 'classic':
            if hasattr(self.model, 'predict_proba'):
                print('Using predict_proba')
                def predictor(texts):
                    # Transform the raw text using the tokenizer
                    text_transformed = self.tokenizer.transform(texts)
                    # Return predicted probabilities
                    return self.model.predict_proba(text_transformed)
            else:
                print('Using predict')
                def predictor(texts):
                    # Transform the raw text using the tokenizer
                    text_transformed = self.tokenizer.transform(texts)
                    # Return predicted class labels
                    return self.model.predict(text_transformed)
        
        elif self.model_type in ['fasttext', 'word2vec', 'glove']:
            if verbose:
                print('Using FastText/Word2Vec/GloVe embeddings')
            def predictor(texts):
                # Transform each text into an embedding vector
                embeddings = []
                for text in texts:
                    words = text.split()
                    vectors = []
                    for word in words:
                        if word in self.tokenizer:  # Si la palabra está en el vocabulario
                            vectors.append(self.tokenizer[word])
                        else:  # Si la palabra no está en el vocabulario
                            if self.model_type == 'glove':
                                vectors.append(np.zeros(100))
                            else:
                                vectors.append(np.zeros(300))  # Usar un vector de ceros
                    if vectors:  # Si hay al menos una palabra en el vocabulario
                        vector = np.mean(vectors, axis=0)
                    else:  # Si no hay palabras en el vocabulario
                        vector = np.zeros(300)  # Asume embeddings de 300 dimensiones
                    embeddings.append(vector)
                embeddings = np.array(embeddings)
                
                # Return predicted probabilities or class labels
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(embeddings)
                else:
                    # Si el modelo no tiene predict_proba, devuelve un array bidimensional con las clases predichas
                    predictions = self.model.predict(embeddings)
                    return np.array([[1 - pred, pred] for pred in predictions])  # Para clasificación binaria
        elif self.model_type == 'transformers':
            self.model.eval()
            device = next(self.model.parameters()).device
            tokenizer = self.tokenizer

            def predictor(texts):
                # Tokenize batch of texts
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert logits to probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                return probabilities.detach().cpu().numpy()
        else:
            raise ValueError("Unsupported model type. Use 'classic' for scikit-learn models.")

        # Get the model's prediction for the sample
        if verbose:
            print('Predicted:', np.argmax(predictor([sample])))

        # Initialize LIME explainer
        if self.model_type == 'transformers':
            explainer = LimeTextExplainer(class_names=self.class_names, bow=self.bow, char_level=False, split_expression=lambda x: x.split(), mask_string=tokenizer.mask_token)
        else:
            explainer = LimeTextExplainer(class_names=self.class_names, bow=self.bow, char_level=False)

        # Explain the model's prediction
        explanation = explainer.explain_instance(sample, predictor, num_features=num_features)

        html_code = explanation.as_html()
        
        if verbose:
            print("Generating HTML code for LIME explanation")

        css_hide_tables = """
        <html>
            <style>
            .lime.explanation {
            display: none !important;
            }

            .lime.predict_proba {
				position: absolute;
				left: -9999px;
            }
            
            ::-webkit-scrollbar {
                display: none;
            }
            </style>
            """

        # remove first html tag
        html_code = html_code.replace('<html>', '')
        html_code = css_hide_tables + html_code
        
        if output_image_path is not None:
            # Try using the helper method first (Selenium)
            self._save_html_as_image(html_code, output_image_path, driver_path)
            
            # Fallback or alternative: imgkit (if preferred, uncomment or add logic)
            # try:
            #     with open(output_image_path + '.html', 'w', encoding='utf-8') as f:
            #         f.write(html_code)
            #     imgkit.from_file(output_image_path + '.html', output_image_path + '.png')
            # except Exception as e:
            #     logger.warning(f"imgkit failed: {e}")

        display(HTML(html_code))


    def get_shap_explanation(self, sample_idx, num_features=None, output_image_path=None, verbose=False, driver_path=None, max_evals=22031):

         # Get the sample and target
        sample = self.dataset.iloc[sample_idx]['text']
        target = self.dataset.iloc[sample_idx]['score']

        if verbose:
            print('Sample:', sample)
            print('Target:', target)

        # Define the predictor function based on the model type
        if self.model_type == 'classic':
            if hasattr(self.model, 'predict_proba'):
                print('Using predict_proba')
                def predictor(texts):
                    # Transform the raw text using the tokenizer
                    text_transformed = self.tokenizer.transform(texts)
                    # Return predicted probabilities
                    probas = self.model.predict_proba(text_transformed)

                    return probas

            else:
                print('Using predict')
                def predictor(texts):
                    # Transform the raw text using the tokenizer
                    text_transformed = self.tokenizer.transform(texts)
                    # Return predicted class labels
                    return self.model.predict(text_transformed)
                
        elif self.model_type in ['fasttext', 'word2vec', 'glove']:
            if verbose:
                print('Using FastText/Word2Vec/glove embeddings')
            def predictor(texts):
                # Transform each text into an embedding vector
                embeddings = []
                for text in texts:
                    words = text.split()
                    vectors = []
                    for word in words:
                        if word in self.tokenizer:  # Si la palabra está en el vocabulario
                            vectors.append(self.tokenizer[word])
                        else:  # Si la palabra no está en el vocabulario
                            if self.model_type == 'glove':
                                vectors.append(np.zeros(100))
                            else:
                                vectors.append(np.zeros(300))
                    if vectors:  # Si hay al menos una palabra en el vocabulario
                        vector = np.mean(vectors, axis=0)
                    else:  # Si no hay palabras en el vocabulario
                        if self.model_type == 'glove':
                            vector = np.zeros(100)
                        else:
                            vector = np.zeros(300)  # Asume embeddings de 300 dimensiones
                    embeddings.append(vector)
                embeddings = np.array(embeddings)
                
                # Return predicted probabilities or class labels
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(embeddings)
                else:
                    # Si el modelo no tiene predict_proba, devuelve un array bidimensional con las clases predichas
                    predictions = self.model.predict(embeddings)
                    return np.array([[1 - pred, pred] for pred in predictions])  # Para clasificación binaria

        elif self.model_type == 'transformers':

            print('Using transformers')

            self.model.eval()

            device = next(self.model.parameters()).device

            def predictor(texts):
                tv = torch.tensor([self.tokenizer.encode(text, max_length=512, truncation=True, padding="max_length") for text in texts]).to(device)
                attention_mask = (tv != 0).float().to(device)
                outputs = self.model(tv, attention_mask=attention_mask)[0].detach().cpu().numpy()

                scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                val = sp.special.logit(scores)

                return val

        else:
            raise ValueError("Unsupported model type. Use 'classic' for scikit-learn models or 'transformers' for BERT.")

        # Get the model's prediction for the sample
        if verbose:
            print('Predicted:', np.argmax(predictor([sample])))

        masker = shap.maskers.Text()

        if self.model_type == 'transformers':
            explainer = shap.Explainer(predictor, self.tokenizer, output_names=self.class_names)
        else:
            explainer = shap.Explainer(predictor, masker, output_names=self.class_names)

        shap_values = explainer([sample])

        html_code = shap.plots.text(shap_values, display=False)
        
        if output_image_path is not None:
            self._save_html_as_image(html_code, output_image_path, driver_path)
        
        display(HTML(html_code))
        
    def visualize_bert_attention(self, sample_idx, layer=None, head=None, output_image_path=None, driver_path=None):
        """
        Visualiza los mecanismos de atención de BERT usando BertViz.
        
        Args:
            text (str): Texto a analizar
            layer (int): Capa específica a visualizar (None para todas)
            head (int): Cabeza de atención específica (None para todas)
        """
        from bertviz import head_view
        from bertviz.transformers_neuron_view import BertModel, BertTokenizer
        from IPython.display import display
        
        sample = self.dataset.iloc[sample_idx]['text']
        target = self.dataset.iloc[sample_idx]['score']
        
        # Codificar el texto
        inputs = self.tokenizer.encode_plus(sample, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        
        # Obtener las salidas del modelo
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        # Configurar la visualización
        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Visualización interactiva
        html = head_view(
            attention,
            tokens,
            html_action='return',
        )
        
        if output_image_path:
            html_content = getattr(html, 'data', html)
            self._save_html_as_image(html_content, output_image_path, driver_path)
        
        # Also save HTML if needed or just display? 
        # The original code saved to "attention.html". I'll keep saving to html if output_path is given, 
        # but maybe append .html
        
        # display(html) # head_view returns an object that displays itself in notebook usually, or we display it.
        # The original code didn't display it, just saved it.
        # But imported display.
        
        display(html)
        
    def integrated_gradients_explanation(self, sample_idx, true_class=1, target_class = 0, steps=50, output_image_path = None, verbose=False, driver_path=None):
        """
        Genera una explicación usando Integrated Gradients y la visualiza con Captum.
        
        Args:
            sample_idx (int): Índice de la muestra en self.dataset.
            true_class (int): Clase verdadera de la muestra (por defecto 1).
            steps (int): Número de pasos en el cálculo de las IG (por defecto 50).
            verbose (bool): Imprimir información adicional (por defecto False).
        """
        import torch
        from captum.attr import IntegratedGradients
        from captum.attr import visualization as viz
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        baseline_token_id = self.tokenizer.pad_token_id 
        sep_token_id = self.tokenizer.sep_token_id 
        cls_token_id = self.tokenizer.cls_token_id 
        
        print("baseline_token_id:", baseline_token_id)
        print("sep_token_id:", sep_token_id)
        print("cls_token_id:", cls_token_id)
        
        # 1. Extraer el texto de la muestra
        text = self.dataset.iloc[sample_idx]['text']
        
        # 2. Tokenizar el texto. Ajusta parámetros como 'truncation=True' según necesites.
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True
        )
        
        input_ids = inputs['input_ids']
        input_ids_zize = input_ids.size(1)
        input_ids = torch.cat([torch.tensor([[cls_token_id]]), input_ids, torch.tensor([[sep_token_id]])], dim=1).to(device)

        attention_mask = inputs['attention_mask'].to(device)

        attention_mask = torch.cat([
            torch.tensor([[1]], device=device), 
            attention_mask, 
            torch.tensor([[1]], device=device)
        ], dim=1)
        
        original_length = input_ids.size(1) - 2
        
        # 3. Definir la función de forward que recibe los embeddings
        def forward_func(input_embeds):
            """
            Función que toma como entrada los embeddings de tokens
            y devuelve la probabilidad de la clase objetivo.
            """
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            # Utilizamos softmax si queremos probabilidades en vez de logits
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[:, target_class]
        
        # 4. Obtener los embeddings de entrada desde la capa de embeddings de BERT
        embedding_layer = self.model.get_input_embeddings()
        input_embeddings = embedding_layer(input_ids)
        
        # 5. Definir la referencia (baseline). Por ejemplo, ceros con la misma forma
        baseline_input_ids = torch.tensor([
            [cls_token_id] + [baseline_token_id] * original_length + [sep_token_id]
        ], device=device)
        
        print(f"input_ids: {input_ids.size()}")
        print(f"baseline_input_ids: {baseline_input_ids.size()}")
        baseline_embeddings = embedding_layer(baseline_input_ids)
        
        # 6. Crear el objeto de Integrated Gradients
        ig = IntegratedGradients(forward_func)
        
        # 7. Calcular las atribuciones (y la medida de convergencia delta)
        attributions, delta = ig.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            n_steps=steps,
            return_convergence_delta=True,
        )
        
        print("Atribuciones:", attributions.size())
        
        # 8. Sumar las atribuciones a lo largo de la dimensión de embeddings 
        #    para obtener una puntuación por cada token
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        
        # normalizar
        attributions_sum = attributions_sum / torch.norm(attributions_sum)
        
        # 9. Convertir los IDs de tokens a tokens legibles
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Clean up special tokens
        clean_tokens = []
        
        for token in tokens:
            word = ""
            decoded_token = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(token), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            clean_tokens.append(decoded_token)

        
        print("tokens:", tokens)
        print("Tokens decoded:", clean_tokens)
        
        
        # 10. Obtener la predicción del modelo sobre la muestra (para mostrarla en la visualización)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.softmax(output.logits, dim=1)
        predicted_class = torch.argmax(preds, dim=1).item()
        predicted_prob = preds[0, predicted_class].item()
        
        print("predicted_class:", predicted_class)
        print("predicted_prob:", predicted_prob)
        
        # 11. Crear un registro de datos de visualización (VisualizationDataRecord)
        #     Captum mostrará en color los tokens más relevantes según su atribución.
        vis_data_records = []
        vis_data_records.append(
                viz.VisualizationDataRecord(
                word_attributions=attributions_sum.detach().cpu().numpy(), 
                pred_prob=float(predicted_prob),                   # Debe ser float
                pred_class=int(predicted_class),                   # Espera un int
                true_class=int(true_class),                      # Espera un int
                attr_class=int(target_class),                      # Espera un int
                attr_score=float(attributions_sum.sum().item()),   # Debe ser float
                raw_input_ids=clean_tokens,                              # Lista de str
                convergence_score=float(delta.item())              # Debe ser float
            )
        )
        
        # 12. Visualizar
        html_code = viz.visualize_text(vis_data_records)
        
        if output_image_path is not None:
            # Handle both string and IPython.display.HTML object
            html_content = getattr(html_code, 'data', html_code)
            self._save_html_as_image(html_content, output_image_path, driver_path)
            
        
        # Si queremos ver información en consola
        if verbose:
            print("Texto original:", text)
            print("Tokens:", tokens)
            print("Atribuciones (token-level):", attributions_sum)
            print("Delta de convergencia:", delta)
            print("Clase predicha:", predicted_class, "| Probabilidad:", predicted_prob)
            