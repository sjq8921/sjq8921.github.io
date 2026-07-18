export const siteConfig = {
  name: "Sun JingQi",
  description:
    "Academic homepage of Sun JingQi — undergraduate at Shandong University of Finance and Economics, working on computer vision.",
  email: "sjq8921@163.com",
  github: "https://github.com/sjq8921",
  scholar: "https://scholar.google.com/"
};

export const navItems = [
  { label: "Home", href: "/" },
  { label: "Publications", href: "/publications" },
  { label: "Blog", href: "/blog" }
];

export type Publication = {
  title: string;
  authors: string[];
  highlight: string;
  venue: string;
  doi?: string;
  links?: { journal?: string; blog?: string };
  tags: string[];
  cover: string;
};

// TODO: replace the placeholder entries below with real publications.
export const publications: Publication[] = [
  {
    title: "Title of Your Selected Publication: Subtitle or Additional Description",
    authors: ["Sun JingQi", "Coauthor A", "Coauthor B"],
    highlight: "Briefly describe the main contribution of this paper in one sentence.",
    venue: "Journal Name 12, 3456 (2026)",
    doi: "10.xxxx/example-doi",
    links: { journal: "#", blog: "#" },
    tags: ["Keyword 1", "Keyword 2", "Keyword 3"],
    cover: "/assets/pub-1.jpg"
  },
  {
    title: "Title of Another Representative Publication",
    authors: ["Sun JingQi", "Coauthor A", "Coauthor B"],
    highlight: "Summarize the research question, method, or key finding in a compact way.",
    venue: "Journal or Conference Name 8, 1234 (2025)",
    doi: "10.xxxx/example-doi",
    links: { journal: "#", blog: "#" },
    tags: ["Keyword 1", "Keyword 2"],
    cover: "/assets/pub-2.jpg"
  },
  {
    title: "Title of a Third Selected Publication",
    authors: ["Sun JingQi", "Coauthor A"],
    highlight: "Add a short explanation of why this work is representative.",
    venue: "Journal or Conference Name 6, 7890 (2024)",
    doi: "10.xxxx/example-doi",
    links: { journal: "#" },
    tags: ["Keyword 1", "Keyword 2"],
    cover: "/assets/pub-3.jpg"
  },
  {
    title: "Title of a Fourth Selected Publication",
    authors: ["Sun JingQi", "Coauthor A", "Coauthor B"],
    highlight: "Use this line to help readers quickly understand the contribution.",
    venue: "Journal or Conference Name 4, 5678 (2023)",
    doi: "10.xxxx/example-doi",
    links: { journal: "#", blog: "#" },
    tags: ["Keyword 1", "Keyword 2", "Keyword 3"],
    cover: "/assets/pub-4.jpg"
  }
];

export type ExperienceItem = {
  time: string;
  title: string;
  link?: string;
};

export const education: ExperienceItem[] = [
  {
    time: "2025–2025",
    title: "中昊芯英（杭州）科技有限公司, Algorithm Intern",
    link: "http://www.zhcltech.com/"
  },
  {
    time: "2023–Present",
    title: "Shandong University of Finance and Economics, undergraduate student"
  }
];

export const awards: string[] = [
  "University First-Class Scholarship (2023–2024)",
  "University First-Class Scholarship (2024–2025)",
  "University Outstanding Student, 2023–2024"
];

export const news: { date?: string; text: string; link?: string }[] = [];
