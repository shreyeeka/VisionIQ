document.addEventListener("DOMContentLoaded", () => {
  const navbar = document.getElementById("navbar");
  const navToggle = document.getElementById("navToggle");
  const navMobilePanel = document.getElementById("navMobilePanel");
  const navLinks = document.getElementById("navLinks");

  if (navbar) {
    window.addEventListener("scroll", () => {
      navbar.classList.toggle("scrolled", window.scrollY > 12);
    }, { passive: true });
  }

  const closeMobileNav = () => {
    navMobilePanel?.classList.remove("open");
    navToggle?.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  };

  if (navToggle && navMobilePanel) {
    navToggle.addEventListener("click", () => {
      const isOpen = navMobilePanel.classList.toggle("open");
      navToggle.classList.toggle("open", isOpen);
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    navLinks?.querySelectorAll("a, button").forEach((el) => {
      el.addEventListener("click", closeMobileNav);
    });
  }

  const path = window.location.pathname.replace(/\/$/, "") || "/";
  const navMap = {
    "/": "home",
    "/upload": "upload",
    "/dashboard": "dashboard",
    "/about": "about",
    "/history": "history",
    "/login": "login",
    "/signup": "signup",
  };
  const activeKey = navMap[path];
  if (activeKey) {
    document.querySelectorAll(`[data-nav="${activeKey}"]`).forEach((el) => {
      el.classList.add("active");
    });
  }

  const revealEls = document.querySelectorAll(".reveal");
  if (revealEls.length && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach((el) => observer.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("visible"));
  }

  document.querySelectorAll("[data-count]").forEach((el) => {
    const target = parseInt(el.dataset.count, 10);
    if (Number.isNaN(target) || target === 0) return;

    const duration = 1200;
    const start = performance.now();

    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(target * eased);
      if (progress < 1) requestAnimationFrame(step);
    };

    const statObserver = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          requestAnimationFrame(step);
          statObserver.disconnect();
        }
      },
      { threshold: 0.5 }
    );
    statObserver.observe(el);
  });

  document.querySelectorAll(".confidence-bar__fill").forEach((bar) => {
    const width = bar.style.width;
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = width; }, 300);
    });
  });
});
