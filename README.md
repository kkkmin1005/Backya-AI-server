## Life Legacy - GDGoC Backya 4조

### 프로젝트 개요
노인들이 AI 챗봇과 대화하며 자신의 자서전을 만들고, 이를 젊은 세대와 공유하며 정보 교류 및 지혜 전달을 돕는 서비스입니다.

### 프로젝트 목표
자서전을 통해, 고령화로 인한 세대 간 갈등 및 격차를 줄이며 책으로 출판하여 노인들의 생계 유지에 도움이 되는것을 목표로 합니다.

### 프로젝트 설명
**자서전 작성**  
1. 카테고리 기반 질문 구성
  - 유아기, 청소년기, 성인기 둥 인생의 주요 단계를 기준으로 질문 카테고리를 구성합니다.
  - 각 카테고리는 5개의 주요 질문으로 구성되어 있습니다.
2. 질문 응답 및 AI 추가 질문
  - 사용자는 각 질문에 대한 답변을 작성하며 자신의 이야기를 공유합니다.
  - 사용자가 입력한 답변을 바탕으로, AI는 추가적인 관련 질문을 생성하여 심도 있는 대화를 유도합니다.
3. 자서전 작성 완료
  - 각 카테고리의 모든 질문에 답변을 완료하면 해당 카테고리의 자서전이 작성됩니다.
  - 모든 질문에 답변을 마치면 삶을 아우르는 자서전이 완성됩니다.

**이미지 생성**
- 독자들이 자서전에 쉽게 접근할 수 있도록, 각 질문의 답변 내용을 이미지로 생성합니다.  
- 최종적으로 자서전은 이미지와 글로 이루어집니다.  

### 주요 기능

- AI 챗봇: gpt4에 프롬프트 엔지니어링을 이용하여, 삶에 관련된 질문을 던짐
- 이미지 생성: 작성된 자서전을 바탕으로 이미지를 생성해줌(stable diffusion 기반 이미지 생성 모델)
- 카툰화: 생성된 이미지를 만화처럼 보이도록 수정해줌(openCV의 필터 이용)
- 자서전 기록: mysql, aws s3를 이용하여 이미지와 작성된 글 저장

### 구현 영상
https://github.com/user-attachments/assets/7d17637e-2de9-4873-916d-96c11443638f

