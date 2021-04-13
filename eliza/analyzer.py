class Analyzer:

    GOOD_EMOTION = ["happy", "neutral", "calm"]
    CONFUSED_EMOTION = ["surprise", "surprised"]
    BAD_EMOTION = ["contempt", "fear", "anger", "angry", "sad", "fearful", "disgust", "sadness", "sad"]

    def __init__(self):
        self.conversation_user = []
        self.speech_emotion_user = []
        self.face_emotion_user = []

    def add_information(self,face_emotion,speech_emotion, nlu_intent):
        self.conversation_user.append(nlu_intent)
        self.speech_emotion_user.append(speech_emotion)
        self.face_emotion_user.append(face_emotion)

    def is_good_emotion(self, face_emotion, speech_emotion):
        return (face_emotion in self.GOOD_EMOTION or face_emotion == None) and (speech_emotion in self.GOOD_EMOTION or speech_emotion == None)

    def is_bad_emotion(self, face_emotion, speech_emotion):
        return face_emotion in self.BAD_EMOTION or speech_emotion in self.BAD_EMOTION

    def is_confused(self, face_emotion, speech_emotion):
        return face_emotion in self.CONFUSED_EMOTION or speech_emotion in self.CONFUSED_EMOTION

    def get_eliza_answer(self):
        max_speech_emotion = max(self.speech_emotion_user,key=self.speech_emotion_user.count)
        max_face_emotion = max(self.face_emotion_user,key=self.face_emotion_user.count)

        #if no conversation for the moment, we analyze only emotion datas
        if(self.conversation_user == [None] * len(self.conversation_user) or self.conversation_user == []):
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                return "Je vois que vous êtes en pleine forme \nC'est bien"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                return "Quelque chose dont vous voulez parler ? \nJ'ai l'impression que quelque chose vous travaille"
            elif self.is_confused(max_face_emotion, max_speech_emotion):
                return "Vous m'avez l'air confus ? \nTout va bien?"
            else : 
                return "Si vous avez quelque chose à me dire \nJe vous écoute"

        last_conversation = self.conversation_user[-1:]
        print(last_conversation)
        if last_conversation != "" and last_conversation !=  "None" :
            return self.find_answer(last_conversation, max_speech_emotion, max_face_emotion)
        else:
            return "Continuez"

    def find_answer(self, last_conversation, max_speech_emotion, max_face_emotion):
        last_speech_emotion = self.speech_emotion_user[-1:]
        last_face_emotion = self.face_emotion_user[-1:]
        answer_eliza = "Continuez"

        if last_conversation == ["interlocuteurPasBien"]:
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Pourtant, vous avez bonne mine \nMais je vous écoute"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Racontez-moi tout"

        if last_conversation == ["interlocuteurBien"]:
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Je suis ravi d'entendre que vous allez bien"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Je n'ai pas l'impression pourtant \nVous pouvez tout me dire, vous savez"

        if last_conversation == ["interlocuteurPasParler"]:
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Très bien, je n'insiste pas alors"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Je comprends que ce soit dur de s'ouvrir \nMais vous pouvez me faire confiance"

        if last_conversation == ["interlocuteurMerci"]:
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Mais je vous en prie \n C'est normal"
            # We adapt the conversation if the user is happy after negative emotions
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion) and self.is_good_emotion(last_face_emotion, last_speech_emotion):
                answer_eliza = "Je suis heureux de constater \nque notre échange vous a fait du bien"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Je suis là pour vous aider"

        if last_conversation == ["interlocuteurDesole"]:
            answer_eliza = "Vous ne me dérangez pas"
        
        if last_conversation == ["sujetDiscussion"]:
            if self.is_good_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Je vois que parler de cela vous rend heure"
            # We adapt the conversation if the user is happy after negative emotions
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion) and self.is_good_emotion(last_face_emotion, last_speech_emotion):
                answer_eliza = "Ah, je vois que parler de cela vous fait du bien \nContinuez"
            elif self.is_bad_emotion(max_face_emotion, max_speech_emotion):
                answer_eliza = "Dites m'en plus à ce sujet \nJe suis là pour vous aider"

        if self.is_confused(last_face_emotion, last_speech_emotion):
            answer_eliza = "Vous m'avez l'air confus ? \nTout va bien?"
        return answer_eliza