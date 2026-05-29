/* ── Auto-assign left/right alternation from document order ── */
document.querySelectorAll('.timeline-item').forEach((item, i) => {
  item.classList.add(i % 2 === 0 ? 'right' : 'left');
});

/* ── Entrance animations via IntersectionObserver ── */
const items = document.querySelectorAll('.timeline-item');
const years = document.querySelectorAll('.timeline-year');

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.12 });

items.forEach(item => observer.observe(item));
years.forEach(year => observer.observe(year));
document.querySelectorAll('.timeline-month').forEach(m => observer.observe(m));

/* ── Fold / unfold misc cards ── */
document.querySelectorAll('.fold-toggle').forEach(toggle => {
  toggle.addEventListener('click', () => {
    const content = toggle.closest('.timeline-content');
    const nowFolded = content.classList.toggle('folded');
    toggle.textContent = nowFolded ? '∨' : '∧';
    toggle.setAttribute('aria-expanded', String(!nowFolded));
  });
});

/* ── Filter buttons ── */
const filterBtns = document.querySelectorAll('.filter-btn');
const timelineMonths = document.querySelectorAll('.timeline-month');
const timelineYears = document.querySelectorAll('.timeline-year');

// Map filter value to tag classes that match
const filterMap = {
  ai: ['tag-ai', 'tag-symolicai'],
  flutter: ['tag-mobile'],
  web: ['tag-web'],
  docs: ['tag-docs'],
  other: ['tag-scripts', 'tag-security', 'tag-systems', 'tag-current'],
};

function setSideClasses(items) {
  let idx = 0;
  items.forEach(item => {
    item.classList.remove('left', 'right');
    item.classList.add(idx % 2 === 0 ? 'right' : 'left');
    idx++;
  });
}

function applyFilter(filter) {
  // Update button states
  filterBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filter);
  });

  // Show/hide items
  const allItems = document.querySelectorAll('.timeline-item');
  allItems.forEach(item => {
    if (filter === 'all') {
      item.style.display = '';
      item.classList.add('visible');
    } else {
      const tagClasses = filterMap[filter] || [];
      const hasTag = tagClasses.some(tc => item.querySelector('.tag.' + tc));
      item.style.display = hasTag ? '' : 'none';
      if (hasTag) item.classList.add('visible');
    }
  });

  // Re-assign left/right based on VISIBLE items only
  const visibleItems = document.querySelectorAll('.timeline-item:not([style*="display: none"])');
  setSideClasses(visibleItems);

  // Show/hide months (only if they have visible items after them)
  timelineMonths.forEach(month => {
    if (filter === 'all') {
      month.style.display = '';
      month.classList.add('visible');
      return;
    }
    let hasVisible = false;
    let el = month.nextElementSibling;
    while (el && !el.classList.contains('timeline-month') && !el.classList.contains('timeline-year')) {
      if (el.classList.contains('timeline-item') && el.style.display !== 'none') {
        hasVisible = true;
        break;
      }
      el = el.nextElementSibling;
    }
    month.style.display = hasVisible ? '' : 'none';
  });

  // Show/hide years (only if they have visible items)
  timelineYears.forEach(year => {
    if (filter === 'all') {
      year.style.display = '';
      year.classList.add('visible');
      return;
    }
    let hasVisible = false;
    let el = year.nextElementSibling;
    while (el && !el.classList.contains('timeline-year')) {
      if (el.classList.contains('timeline-item') && el.style.display !== 'none') {
        hasVisible = true;
        break;
      }
      el = el.nextElementSibling;
    }
    year.style.display = hasVisible ? '' : 'none';
  });
}

filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    applyFilter(btn.dataset.filter);
  });
});

/* ── Scroll-to-top button ── */
const topBtn = document.getElementById('back-to-top');

window.addEventListener('scroll', () => {
  topBtn.classList.toggle('show', window.scrollY > 320);
}, { passive: true });

topBtn.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});