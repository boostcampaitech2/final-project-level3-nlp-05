setwd("~/Developer")

# ===== Pre-processing =====

load_data <- function(path) {
  df <- read.csv(path)
  head(df)
  
  colnames(df)
  colnames(df) <- c("q1", "empty", "q2", "q3", "q9", "phone_num", "q5", "q6", "q7", "q8", "q10", "q4", "gender", "age", "timestamp", "token")
  df <- df[, c("q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "gender", "age")]
  
  answers_a <- c(
    "남원시는 연말연시를 맞아 남원예촌 예루원마당에서 개최하고 코로나 백신100 scene 등 풍성한 볼거리로 코로나로 지친 이들을 위로할 예정이며, 양인환 남원시 관광과장은 남원예촌의 명품한옥과 어우러지는 겨울경관 조성과 함께 시민들의 소망을 표현할 수 있는 체험형 전시프로그램을 통해 시민들에게 다채로운 볼거리를 제공할 수 있는 계기를 만들도록 하겠다고 말했다.",
    "대성에너지주는 지난 16일 여성가족부로부터 가족친화기업 재인증을 획득했다고 17일 밝혔는데, 가족친화기업 인증은 근로자가 일과 가정생활을 조화롭게 병행할 수 있도록 다양한 일‧가정 양립 제도와 사업을 시행하는 기업에게 심사를 통해 부여되며 각종 인센티브도 함께 주어진다.",
    "SK온이 17일 오전 이사회와 임시주주총회를 열고 최재원 SK그룹 수석부회장을 사내이사 및 각자 대표이사로 선임하기로 의결했으며 최 수석부회장은 이날부터 지동섭 대표이사 사장과 함께 SK온 각자 대표직을 수행하게 된다.",
    "현대 현대자동차그룹이 대내외 급격한 경영 환경에 대응하고 미래의 지속가능한 선순환 체계를 구축할 리더십 확보를 위해 2021년 하반기 임원 인사를 실시했다고 17일 밝혔는데, 이번 인사는 신속한 사업 포트폴리오 전환 및 인적 경쟁력 제고를 위한 변화와 혁신의 방향성을 제시한 것이 핵심이다.",
    "정부는 매출이 감소한 320만명의 소상공인을 대상으로 100만원 상당의 방역지원금을 신규 지원하기로 했으며 이런 내용 등을 담은 방역지원금 및 손실보상지원 확대 방안을 오늘17일 발표했다.",
    "17일 영화 킹스맨 퍼스트 에이전트의 화상 기자 간담회에서 매튜 본 감독은 앞선 시리즈와의 차별점에 대해 배우 랄프 파인즈를 꼽으며 한국과 한국 팬들에 대한 남다른 애정을 표했다.",
    "김경남 소속사 제이알이엔티는 배우 김경남 측이 측간 소음으로 이웃에게 피해를 준 데 대해 17일 공식입장을 통해 먼저 좋지 않은 일로 심려를 끼쳐 죄송하다고 사과했고, 이날 저녁 김경남 배우가 당사자 분을 찾아가 이야기를 나눴다며 진심으로 사과드리고 앞으로는 더 주의하겠다고 말씀드렸다고 전했다.",
    "유엔총회는 16일 미국 뉴욕 유엔본부에서 본회의를 열어 북한의 조직적이고 광범위한 인권침해를 규탄하고 코로나19 백신 협력을 당부하는 북한인권결의안을 표결 없이 컨센서스로 채택하여 17년 연속으로 유엔총회를 통과했다.",
    "국방부 보통군사법원은 17일 군인 등 강제추행치상 등의 혐의로 구속기소된 공군 장 중사에게 징역 9년을 선고했다.",
    "파이낸셜뉴스 윤석열 국민의힘 대선 후보는 17일 페이스북을 통해 당선 즉시 흉악 범죄와의 전쟁에서 반드시 승리하겠다고 밝혔는데, 윤 후보는 당선 즉시 흉악 범죄와의 전쟁을 선포하겠다며 26년간 검사로서 형사법집행을 해온 전문가로서 제가 국민의 안전을 확실히 지키겠다고 강조했다."
  )
  
  # requires manually checking any flips
  df$q1 <- as.factor(ifelse(df$q1 == answers_a[1], 1, 2))
  df$q2 <- as.factor(ifelse(df$q2 == answers_a[2], 1, 2))
  df$q3 <- as.factor(ifelse(df$q3 == answers_a[3], 1, 2))
  df$q4 <- as.factor(ifelse(df$q4 == answers_a[4], 1, 2))
  df$q5 <- as.factor(ifelse(df$q5 == answers_a[5], 1, 2))
  df$q6 <- as.factor(ifelse(df$q6 == answers_a[6], 1, 2))
  df$q7 <- as.factor(ifelse(df$q7 == answers_a[7], 1, 2))
  df$q8 <- as.factor(ifelse(df$q8 == answers_a[8], 1, 2))
  df$q9 <- as.factor(ifelse(df$q9 == answers_a[9], 1, 2))
  df$q10 <- as.factor(ifelse(df$q10 == answers_a[10], 1, 2))
  
  # need to manually fill-in initial missing values
  df$age[1:3] <- c("20대", "30대", "20대")
  df$gender[1:3] <- c("남자", "남자", "여자")
  
  df$age <- factor(df$age, levels=c("10대", "20대", "30대", "40대", "50대 이상"))
  df$gender <- factor(df$gender, levels=c("남자", "여자"))
  levels(df$age) <- c("10", "20", "30", "40", "50")
  levels(df$gender) <- c("m", "f")
  
  head(df)
  
  write.csv(df, file="research_cleansed.csv")
  
  return (df)
}

df = load_data("top-k_research.csv")

# ===== Subsetting =====

# by age group
df <- df[df$age == "20",]
df <- df[df$age == "50",]

# by gender
df <- df[df$gender == "m",]
df <- df[df$gender == "f",]


# ===== Plotting =====

plot_data <- function(df) {
  par(mfrow=c(3, 4))
  plot(df$q1)
  plot(df$q2)
  plot(df$q3)
  plot(df$q4)
  plot(df$q5)
  plot(df$q6)
  plot(df$q7)
  plot(df$q8)
  plot(df$q9)
  plot(df$q10)
  plot(df$age)
  plot(df$gender)
}

plot_data(df)

# ===== Similar =====

similar_scheme <- read.csv("similar_coding_scheme.csv")

top3 <- rep(0, nrow(df))
top5 <- rep(0, nrow(df))
top7 <- rep(0, nrow(df))
orig <- rep(0, nrow(df))

scores <- data.frame(top3, top5, top7, orig)

for (i in 1:nrow(df)) {
  for (j in 1:10) {
    if (df[i, j] == 1) {
      scores[i, 1:4] <- scores[i, 1:4] + similar_scheme[j, 2:5]
    } else {
      scores[i, 1:4] <- scores[i, 1:4] + similar_scheme[j, 6:9]
    }
  }
}
head(scores)
friedman.test(as.matrix(scores))
colSums(scores)

rowSums(apply(scores, 1, rank))

quade.test(as.matrix(scores))

# ===== Length =====

length_scheme <- read.csv("length_coding_scheme.csv")

long <- rep(0, nrow(df))
scores <- data.frame(long)

total_long = 0

for (i in 1:nrow(df)) {
  for (j in 1:10) {
    if (df[i, j] == 1) {
      scores[i, 1] <- scores[i, 1] + length_scheme[j, 2]
      total_long <- total_long + length_scheme[j, 2]
    } else {
      scores[i, 1] <- scores[i, 1] + 1 - length_scheme[j, 2]
      total_long <- total_long + 1 - length_scheme[j, 2]
    }
  }
}
par(mfrow=c(1, 1))
hist(scores$long, breaks=0:10)

binom.test(total_long, nrow(df)*10, p=0.5)
