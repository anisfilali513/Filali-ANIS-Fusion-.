"""
╔════════════════════════════════════════════════════════════════════════════════╗
║  IMPERIAL FUSION SYSTEM v2.0 - SOVEREIGN HYBRID ANALYSIS ENGINE              ║
║  =========================================================================== ║
║  ARCHITECTURE: STREAMLIT × NEURAL EMBEDDINGS × STRUCTURAL SEMIOTICS × FILALI ║
║  DEVELOPER: ANIS FILALI - COMPARATIVE MYTHOLOGY & AI FUSION LAB             ║
║  OUTPUT: COMPREHENSIVE MYTHOLOGICAL ANALYSIS WITH SOVEREIGN INSIGHTS         ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum
from transformers import AutoTokenizer, AutoModel
from nltk.stem.isri import ISRIStemmer
import json
from datetime import datetime

# ==========================================
# 1. تعريف أنواع الحضور السيادية
# ==========================================
class SovereignPresence(Enum):
    """أنواع الحضور السيادي (الفيلالية المعززة)"""
    IMPERIAL_AFFIRMATIVE = "حضور إمبراطوري إثباتي"
    SOVEREIGN_NEGATED = "حضور سيادي منفي"
    THRONAL_CONSTRUCTIVE = "حضور عروشي بنائي"  
    DESTRUCTIVE_ABSENCE = "غياب هدّام"
    AMBIGUOUS_SOVEREIGNTY = "سيادة غامضة"
    ETERNAL_PRESENCE = "حضور أبدي"

# ==========================================
# 2. كائن التحليل السيادي
# ==========================================
@dataclass
class ImperialAnalysis:
    """تحليل إمبراطوري شامل"""
    text: str
    genre: str
    timestamp: str
    
    # التحليل العصبي
    neural_embedding: np.ndarray
    neural_magnitude: float
    neural_confidence: float
    
    # التحليل البنيوي
    structural_zscore: float
    depth_score: float
    presence_type: SovereignPresence
    
    # التحليل الهجين
    hybrid_score: float
    fusion_confidence: float
    
    # النتائج التفصيلية
    detected_myths: List[Dict] = field(default_factory=list)
    semantic_categories: Dict = field(default_factory=dict)
    presence_patterns: Dict = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def sovereignty_level(self) -> str:
        """مستوى السيادة بناءً على النتائج الهجينة"""
        if self.hybrid_score >= 0.8:
            return "إمبراطورية عالية"
        elif self.hybrid_score >= 0.6:
            return "سيادية متوسطة"
        elif self.hybrid_score >= 0.4:
            return "سيادة محدودة"
        else:
            return "سيادة ضعيفة"

# ==========================================
# 3. المحرك الهجين السيادي (Sovereign Hybrid)
# ==========================================
class ImperialFusionEngine:
    """المحرك الإمبراطوري الهجين: يدمج Streamlit + Neural + Structural"""
    
    def __init__(self):
        # تحميل المحركات الأساسية
        self._load_neural_engine()
        self._load_structural_engine()
        self._load_mythological_database()
        
        # إعدادات السيادة
        self.sovereignty_weights = {
            'neural': 0.35,
            'structural': 0.30,
            'mythological': 0.20,
            'contextual': 0.15
        }
        
    def _load_neural_engine(self):
        """تحميل المحرك العصبي (AMARA)"""
        with st.spinner("🔄 جاري إيقاظ القوى العصبية..."):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
                self.model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
                st.success("✅ المحرك العصبي جاهز")
            except:
                st.warning("⚠️ المحرك العصبي غير متوفر، استخدام وضع محاكاة")
                self.tokenizer = None
                self.model = None
    
    def _load_structural_engine(self):
        """تحميل المحرك البنيوي (Filali)"""
        self.stemmer = ISRIStemmer()
        
        # القواميس السيادية المعززة
        self.imperial_schema = {
            "العروش": ["عرش", "مملكة", "إمبراطورية", "تاج", "صولجان"],
            "الخلود": ["أبدي", "سرمدي", "خالد", "دائم", "لا يفنى"],
            "القوة": ["سلطان", "هيمنة", "سيطرة", "قهر", "غلبة"],
            "الضعف": ["زوال", "انحلال", "تفكك", "انهيار", "ضعف"],
            "البناء": ["تشييد", "تأسيس", "إنشاء", "برج", "صرح"],
            "الهدم": ["هدم", "تحطيم", "تدمير", "إبادة", "محو"]
        }
        
        # أدوات النفي والتركيز المعززة
        self.negators = ["لم", "لن", "لا", "ليس", "غير", "ما", "إنْ", "لما"]
        self.intensifiers = ["بالتأكيد", "قطعاً", "بدون شك", "يقيناً", "حتماً"]
        
        st.success("✅ المحرك البنيوي جاهز")
    
    def _load_mythological_database(self):
        """تحميل قاعدة البيانات الأسطورية"""
        self.myths_database = {
            "جلجامش": {
                "arabic_name": "جلجامش",
                "keywords": ["جلجامش", "اوروك", "انكيدو", "الخلود", "الموت"],
                "category": "البطولة",
                "power_level": 0.9
            },
            "عشتار": {
                "arabic_name": "عشتار",
                "keywords": ["عشتار", "انانا", "الحب", "الحرب", "الخصوبة"],
                "category": "الآلهة",
                "power_level": 0.8
            },
            "أودين": {
                "arabic_name": "أودين",
                "keywords": ["أودين", "فالهالا", "المجد", "الحكمة", "التضحية"],
                "category": "الحكمة",
                "power_level": 0.85
            }
        }
        
    def analyze_imperial(self, text: str, genre: str = "Mythic (أسطوري)") -> ImperialAnalysis:
        """
        التحليل الإمبراطوري الشامل للنص
        
        Args:
            text: النص المدخل
            genre: النوع الأدبي
            
        Returns:
            ImperialAnalysis: نتائج التحليل الكاملة
        """
        
        # إنشاء شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # === المرحلة 1: التحليل العصبي ===
        status_text.text("🧠 المرحلة العصبية: استخراج المعالم العميقة...")
        neural_results = self._analyze_neural(text)
        progress_bar.progress(25)
        
        # === المرحلة 2: التحليل البنيوي ===
        status_text.text("🏛️ المرحلة البنيوية: تحليل الهيكل السيادي...")
        structural_results = self._analyze_structural(text, genre)
        progress_bar.progress(50)
        
        # === المرحلة 3: اكتشاف الأساطير ===
        status_text.text("🔍 المرحلة الأسطورية: البحث عن الرموز...")
        myth_results = self._detect_myths(text)
        progress_bar.progress(75)
        
        # === المرحلة 4: الاندماج السيادي ===
        status_text.text("⚡ المرحلة الإمبراطورية: دمج القوى...")
        fusion_results = self._fuse_imperial(
            neural_results, 
            structural_results, 
            myth_results, 
            text
        )
        progress_bar.progress(100)
        
        status_text.text("✅ التحليل الإمبراطوري اكتمل!")
        
        return fusion_results
    
    def _analyze_neural(self, text: str) -> Dict:
        """التحليل العصبي للنص"""
        if self.tokenizer and self.model:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # استخراج المميزات
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                magnitude = np.linalg.norm(embedding)
                
                # حساب الثقة العصبية
                confidence = min(1.0, magnitude / 10)  # تطبيع
                
                return {
                    "embedding": embedding,
                    "magnitude": float(magnitude),
                    "confidence": float(confidence),
                    "dimensions": embedding.shape[0],
                    "status": "success"
                }
            except Exception as e:
                st.error(f"خطأ في التحليل العصبي: {e}")
        
        # وضع المحاكاة
        return {
            "embedding": np.random.randn(768),
            "magnitude": np.random.uniform(5, 10),
            "confidence": np.random.uniform(0.6, 0.9),
            "dimensions": 768,
            "status": "simulated"
        }
    
    def _analyze_structural(self, text: str, genre: str) -> Dict:
        """التحليل البنيوي السيادي للنص"""
        tokens = text.split()
        roots = [self.stemmer.stem(t) for t in tokens]
        
        score = 0.0
        category_scores = {cat: 0 for cat in self.imperial_schema.keys()}
        evidence = []
        
        # تحليل كل كلمة في سياقها
        for i, (token, root) in enumerate(zip(tokens, roots)):
            # البحث في الفئات السيادية
            for category, keywords in self.imperial_schema.items():
                if any(keyword in token or keyword in root for keyword in keywords):
                    category_scores[category] += 1
                    
                    # تحليل السياق
                    context_start = max(0, i-3)
                    context_end = min(len(tokens), i+4)
                    context = tokens[context_start:context_end]
                    
                    # تأثير النفي
                    if any(neg in context for neg in self.negators):
                        score -= 2.0
                        evidence.append(f"نفي '{token}' في سياق {category}")
                    # تأثير التأكيد
                    elif any(intense in context for intense in self.intensifiers):
                        score += 2.5
                        evidence.append(f"تأكيد '{token}' في سياق {category}")
                    else:
                        score += 1.5
                        evidence.append(f"وجود '{token}' في سياق {category}")
        
        # تحديد نوع الحضور
        presence_type = self._determine_presence_type(score, category_scores)
        
        # حساب عمق التحليل
        depth_score = min(10.0, abs(score) / len(tokens) * 100) if tokens else 0
        
        return {
            "zscore": float(score),
            "depth_score": float(depth_score),
            "presence_type": presence_type,
            "category_scores": category_scores,
            "evidence": evidence[:10],  # أول 10 أدلة فقط
            "token_count": len(tokens)
        }
    
    def _determine_presence_type(self, score: float, category_scores: Dict) -> SovereignPresence:
        """تحديد نوع الحضور السيادي"""
        
        # تحليل توزيع الفئات
        positive_categories = ["العروش", "الخلود", "القوة", "البناء"]
        negative_categories = ["الضعف", "الهدم"]
        
        pos_score = sum(category_scores.get(cat, 0) for cat in positive_categories)
        neg_score = sum(category_scores.get(cat, 0) for cat in negative_categories)
        
        if score > 5 and pos_score > neg_score * 2:
            return SovereignPresence.IMPERIAL_AFFIRMATIVE
        elif score > 3 and category_scores.get("البناء", 0) > 2:
            return SovereignPresence.THRONAL_CONSTRUCTIVE
        elif score < -3 and category_scores.get("الهدم", 0) > 2:
            return SovereignPresence.DESTRUCTIVE_ABSENCE
        elif score < 0 and any(neg in self.negators for neg in ["لم", "لن", "ليس"]):
            return SovereignPresence.SOVEREIGN_NEGATED
        elif pos_score == 0 and neg_score == 0:
            return SovereignPresence.AMBIGUOUS_SOVEREIGNTY
        else:
            return SovereignPresence.ETERNAL_PRESENCE
    
    def _detect_myths(self, text: str) -> List[Dict]:
        """اكتشاف الأساطير في النص"""
        detected = []
        
        for myth_name, myth_data in self.myths_database.items():
            evidence = []
            keyword_count = 0
            
            # البحث عن الكلمات المفتاحية
            for keyword in myth_data["keywords"]:
                count = text.lower().count(keyword.lower())
                if count > 0:
                    keyword_count += count
                    evidence.append(f"كلمة '{keyword}' ظهرت {count} مرة")
            
            if keyword_count > 0:
                # حساب درجة الحضور
                presence_score = min(1.0, keyword_count / len(text.split()) * 100)
                
                detected.append({
                    "name": myth_data["arabic_name"],
                    "original_name": myth_name,
                    "category": myth_data["category"],
                    "presence_score": presence_score,
                    "evidence": evidence,
                    "power_level": myth_data["power_level"],
                    "keyword_count": keyword_count
                })
        
        # ترتيب حسب قوة الحضور
        detected.sort(key=lambda x: x["presence_score"], reverse=True)
        return detected
    
    def _fuse_imperial(self, neural: Dict, structural: Dict, myths: List, text: str) -> ImperialAnalysis:
        """دمج كل التحليلات في تحليل إمبراطوري موحد"""
        
        # حساب النتيجة الهجينة
        neural_component = neural["confidence"] * self.sovereignty_weights['neural']
        structural_component = (structural["zscore"] + 10) / 20 * self.sovereignty_weights['structural']
        
        # حساب المكون الأسطوري
        myth_component = 0
        if myths:
            avg_myth_score = sum(m["presence_score"] for m in myths) / len(myths)
            myth_component = avg_myth_score * self.sovereignty_weights['mythological']
        
        # حساب المكون السياقي
        contextual_component = min(1.0, len(text.split()) / 500) * self.sovereignty_weights['contextual']
        
        # النتيجة النهائية
        hybrid_score = neural_component + structural_component + myth_component + contextual_component
        
        # توليد الرؤى
        insights = self._generate_insights(neural, structural, myths, hybrid_score)
        
        # توليد التوصيات
        recommendations = self._generate_recommendations(hybrid_score, structural["presence_type"], myths)
        
        return ImperialAnalysis(
            text=text,
            genre="Mythic (أسطوري)",
            timestamp=datetime.now().isoformat(),
            neural_embedding=neural.get("embedding", np.zeros(768)),
            neural_magnitude=neural["magnitude"],
            neural_confidence=neural["confidence"],
            structural_zscore=structural["zscore"],
            depth_score=structural["depth_score"],
            presence_type=structural["presence_type"],
            hybrid_score=hybrid_score,
            fusion_confidence=min(1.0, hybrid_score * 0.8 + neural["confidence"] * 0.2),
            detected_myths=myths,
            semantic_categories=structural["category_scores"],
            presence_patterns={"structural": structural["evidence"][:5]},
            insights=insights,
            recommendations=recommendations
        )
    
    def _generate_insights(self, neural: Dict, structural: Dict, myths: List, hybrid_score: float) -> List[str]:
        """توليد رؤى عميقة من التحليل"""
        insights = []
        
        # رؤى عصبية
        if neural["magnitude"] > 8:
            insights.append("🔮 النص يحمل كثافة دلالية عالية تشير إلى عمق أسطوري")
        elif neural["magnitude"] < 5:
            insights.append("🌫️ الكثافة الدلالية منخفضة، النص قد يكون سطحيًا")
        
        # رؤى بنيوية
        if structural["zscore"] > 3:
            insights.append("🏰 البناء السيادي قوي ومتماسك")
        elif structural["zscore"] < -2:
            insights.append("⚰️ هناك توجه هدمي أو تفكيكي في النص")
        
        # رؤى أسطورية
        if myths:
            top_myth = myths[0]["name"]
            insights.append(f"🧝 الرئيسية '{top_myth}' تهيمن على النسيج النصي")
        
        # رؤى هجينة
        if hybrid_score > 0.7:
            insights.append("👑 الاندماج السيادي ناجح: النص يحمل بصمة إمبراطورية")
        elif hybrid_score < 0.3:
            insights.append("🕸️ الاندماج ضعيف: السيادة مشتتة أو غائبة")
        
        return insights
    
    def _generate_recommendations(self, hybrid_score: float, presence_type: SovereignPresence, myths: List) -> List[str]:
        """توليد توصيات سيادية"""
        recommendations = []
        
        # بناءً على النتيجة الهجينة
        if hybrid_score < 0.4:
            recommendations.append("📈 اقترح إضافة تعابير سيادية لتعزيز الحضور الإمبراطوري")
        
        # بناءً على نوع الحضور
        if presence_type == SovereignPresence.DESTRUCTIVE_ABSENCE:
            recommendations.append("⚠️ احذر من الهيمنة الهدمية، وازن بالبناء")
        elif presence_type == SovereignPresence.IMPERIAL_AFFIRMATIVE:
            recommendations.append("✅ الحضور الإثباتي قوي، يمكن البناء عليه لتأسيس رواية")
        
        # بناءً على الأساطير
        if len(myths) > 3:
            recommendations.append("🔗 هناك تعدد أسطوري، يمكن استثماره في بناء عالم موازٍ")
        
        return recommendations

# ==========================================
# 4. واجهة المستخدم الإمبراطورية (Streamlit)
# ==========================================
class ImperialInterface:
    """الواجهة الإمبراطورية للتحليل الهجين"""
    
    def __init__(self):
        self.engine = ImperialFusionEngine()
        self._setup_ui()
    
    def _setup_ui(self):
        """إعداد واجهة المستخدم"""
        st.set_page_config(
            page_title="الإمبراطورية الهجينة - Filali-AMARA Fusion",
            page_icon="👑",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS مخصص
        st.markdown("""
            <style>
            .imperial-title {
                text-align: center;
                color: #D4AF37;
                font-size: 3em;
                margin-bottom: 20px;
            }
            .sovereign-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;
            }
            .myth-card {
                border-left: 5px solid #D4AF37;
                padding: 10px;
                margin: 5px 0;
                background: #f9f9f9;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def render(self):
        """عرض الواجهة الرئيسية"""
        
        # العنوان الإمبراطوري
        st.markdown('<h1 class="imperial-title">👑 الإمبراطورية الهجينة</h1>', unsafe_allow_html=True)
        st.markdown("### Filali-AMARA Fusion System v2.0 - التحليل السيادي المتكامل")
        
        # شريط جانبي للمعلومات
        with st.sidebar:
            st.markdown("## 🏛️ معلومات النظام")
            st.info("""
            **المحركات النشطة:**
            - 🧠 AMARA العصبي
            - 🏛️ Filali البنيوي  
            - 👑 الاندماج السيادي
            """)
            
            st.markdown("## 📊 إحصائيات")
            if 'last_analysis' in st.session_state:
                analysis = st.session_state.last_analysis
                st.metric("النتيجة الهجينة", f"{analysis.hybrid_score:.2%}")
                st.metric("مستوى السيادة", analysis.sovereignty_level)
        
        # منطقة الإدخال الرئيسية
        col1, col2 = st.columns([3, 1])
        
        with col1:
            input_text = st.text_area(
                "أدخل النص الإمبراطوري للتحليل:",
                height=300,
                placeholder="""مثال: جلجامش، ملك أوروك العظيم، لم يرضَ بمصير البشر...
أودين، سيد الآلهة، ضحى بعينيه للحكمة...
عشتار، إلهة الحب والخصوبة، تنزل إلى العالم السفلي..."""
            )
        
        with col2:
            genre = st.selectbox(
                "النوع الأدبي:",
                ["Mythic (أسطوري)", "Epic (ملحمي)", "Classic (كلاسيكي)", "Modern (حداثي)"]
            )
            
            analysis_depth = st.select_slider(
                "عمق التحليل:",
                options=["سطحي", "متوسط", "عميق", "شامل"]
            )
            
            if st.button("🚀 إطلاق التحليل الإمبراطوري", type="primary", use_container_width=True):
                if input_text.strip():
                    with st.spinner("⚡ جاري التحليل الإمبراطوري..."):
                        analysis = self.engine.analyze_imperial(input_text, genre)
                        st.session_state.last_analysis = analysis
                        self._display_results(analysis)
                else:
                    st.warning("⚠️ الرجاء إدخال نص للتحليل")
        
        # عرض مثال إن لم يكن هناك نص
        if not input_text and 'last_analysis' not in st.session_state:
            self._display_example()
    
    def _display_results(self, analysis: ImperialAnalysis):
        """عرض نتائج التحليل"""
        
        st.divider()
        st.markdown("## 📜 التقرير الإمبراطوري")
        
        # بطاقات النتائج الرئيسية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="sovereign-card">', unsafe_allow_html=True)
            st.metric("النتيجة الهجينة", f"{analysis.hybrid_score:.2%}")
            st.caption(f"مستوى: {analysis.sovereignty_level}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="sovereign-card">', unsafe_allow_html=True)
            st.metric("نوع الحضور", analysis.presence_type.value)
            st.caption("التصنيف السيادي")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="sovereign-card">', unsafe_allow_html=True)
            st.metric("القوة العصبية", f"{analysis.neural_magnitude:.2f}")
            st.caption(f"ثقة: {analysis.neural_confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="sovereign-card">', unsafe_allow_html=True)
            st.metric("العمق البنيوي", f"{analysis.depth_score:.2f}")
            st.caption(f"Z-Score: {analysis.structural_zscore:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # === القسم 1: الأساطير المكتشفة ===
        st.markdown("## 🏺 الأساطير المكتشفة")
        
        if analysis.detected_myths:
            for myth in analysis.detected_myths[:5]:  # أول 5 فقط
                with st.expander(f"🧝 {myth['name']} - حضور {myth['presence_score']:.2%}"):
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.write(f"**الفئة:** {myth['category']}")
                        st.write(f"**قوة الأسطورة:** {myth['power_level']:.2f}")
                    with cols[1]:
                        st.metric("عدد الكلمات", myth['keyword_count'])
                    with cols[2]:
                        st.metric("درجة الحضور", f"{myth['presence_score']:.2%}")
                    
                    if myth['evidence']:
                        st.write("**الأدلة:**")
                        for evidence in myth['evidence'][:3]:
                            st.write(f"- {evidence}")
        else:
            st.info("لم يتم اكتشاف أساطير رئيسية في النص")
        
        # === القسم 2: التصورات البصرية ===
        st.markdown("## 📊 التصورات الإمبراطورية")
        
        # الرسم البياني 1: توزيع الفئات السيميائية
        if analysis.semantic_categories:
            fig1 = px.bar(
                x=list(analysis.semantic_categories.keys()),
                y=list(analysis.semantic_categories.values()),
                title="توزيع الفئات السيميائية في النص",
                color=list(analysis.semantic_categories.values()),
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # الرسم البياني 2: تدفق الحضور الأسطوري
        fig2 = go.Figure(data=go.Scatterpolar(
            r=[
                analysis.hybrid_score * 100,
                analysis.neural_confidence * 100,
                analysis.depth_score * 10,
                len(analysis.detected_myths) * 20
            ],
            theta=['الهجين', 'العصبية', 'العمق', 'التنوع'],
            fill='toself',
            name='مقاييس السيادة'
        ))
        
        fig2.update_layout(
            title='مخطط السيادة الإمبراطورية',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # === القسم 3: الرؤى والتوصيات ===
        st.markdown("## 💡 الرؤى والتوصيات السيادية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 الرؤى التحليلية")
            for i, insight in enumerate(analysis.insights, 1):
                st.info(f"{i}. {insight}")
        
        with col2:
            st.markdown("### 🎯 التوصيات الإمبراطورية")
            for i, recommendation in enumerate(analysis.recommendations, 1):
                st.success(f"{i}. {recommendation}")
        
        # === القسم 4: التفاصيل التقنية ===
        with st.expander("🔧 التفاصيل التقنية (للمحللين)"):
            st.json({
                "metadata": {
                    "timestamp": analysis.timestamp,
                    "genre": analysis.genre,
                    "text_length": len(analysis.text),
                    "word_count": len(analysis.text.split())
                },
                "neural_analysis": {
                    "embedding_dimensions": analysis.neural_embedding.shape[0],
                    "magnitude": analysis.neural_magnitude,
                    "confidence": analysis.neural_confidence
                },
                "structural_analysis": {
                    "zscore": analysis.structural_zscore,
                    "depth": analysis.depth_score,
                    "presence_type": analysis.presence_type.value
                },
                "fusion_analysis": {
                    "hybrid_score": analysis.hybrid_score,
                    "fusion_confidence": analysis.fusion_confidence,
                    "sovereignty_level": analysis.sovereignty_level
                }
            })
    
    def _display_example(self):
        """عرض مثال توضيحي"""
        st.divider()
        st.markdown("## 📖 مثال توضيحي")
        
        example_text = """
        جلجامش، ملك أوروك العظيم، لم يرضَ بمصير البشر من الموت والفناء.
        سار في رحلته الطويلة باحثاً عن سر الخلود، متحدياً الآلهة ومتخطياً المخاطر.
        لكنه في النهاية أدرك أن الخلود ليس في الحياة الأبدية، بل في الأعمال الخالدة
        التي تترك أثراً في ذاكرة البشر. فعاد إلى مدينته يحمل حكمة جديدة:
        أن الموت حقيقة لا مفر منها، لكن الذكرى تبقى خالدة.
        
        أودين، سيد الآلهة في الأساطير الإسكندنافية، ضحى بعينيه للحصول على الحكمة
        من بئر ميمير. لقد اختار المعرفة على البصر، والفهم على القوة العمياء.
        هذه التضحية جعلته إلهاً للحكمة والمعرفة، وليس فقط إلهاً للحرب والموت.
        
        عشتار، إلهة الحب والخصوبة في الأساطير البابلية، نزلت إلى العالم السفلي
        لمواجهة أختها إيرشكيغال. في رحلتها هذه، واجهت الموت والانبعاث،
        لتعود بفهم أعمق لدورة الحياة والموت والخصوبة.
        """
        
        st.info("💡 **جرب هذا النص التوضيحي:**")
        st.code(example_text, language="arabic")
        
        if st.button("🔬 تحليل النص التوضيحي"):
            with st.spinner("⚡ جاري التحليل الإمبراطوري للنص التوضيحي..."):
                analysis = self.engine.analyze_imperial(example_text, "Mythic (أسطوري)")
                st.session_state.last_analysis = analysis
                self._display_results(analysis)

# ==========================================
# 5. الدالة الرئيسية للتشغيل
# ==========================================
def main():
    """الدالة الرئيسية لتشغيل النظام الإمبراطوري"""
    
    # تهيئة الواجهة
    interface = ImperialInterface()
    
    # عرض الواجهة
    interface.render()
    
    # تذييل الصفحة
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("👑 الإمبراطورية الهجينة v2.0")
    
    with col2:
        st.caption("🧠 AMARA × 🏛️ Filali Fusion")
    
    with col3:
        st.caption("© 2024 Comparative Mythology & AI Fusion Lab")

# ==========================================
# 6. نقطة الدخول الرئيسية
# ==========================================
if __name__ == "__main__":
    # إعدادات أولية
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # تشغيل التطبيق
    main()
