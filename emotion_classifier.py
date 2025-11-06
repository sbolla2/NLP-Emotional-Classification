class EmotionExample: 
    """
    Represents a single piece of training or testing data
    """
    def __init__(self, tokens: list, transformer_tokens: dict[str, list[int]] | None, emotional_intensity: float, emotional_polarity: float, empathy: float):
        #id,article_id,conversation_id,turn_id,"speaker_id","text",person_id,person_id_1,person_id_2,Emotion,EmotionalPolarity,Empathy
        self.tokens = tokens
        self.transformer_tokens = transformer_tokens
        self.emotional_intensity = emotional_intensity
        self.emotional_polarity = emotional_polarity
        self.empathy = empathy