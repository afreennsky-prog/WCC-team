document.addEventListener('DOMContentLoaded', () => {
  const queueRows = document.querySelectorAll('.queue-row');
  const miniBars = document.querySelectorAll('.mini-bar span');

  queueRows.forEach((row, index) => {
    row.style.opacity = '0';
    row.style.transform = 'translateY(10px)';

    setTimeout(() => {
      row.style.transition = 'all 0.5s ease';
      row.style.opacity = '1';
      row.style.transform = 'translateY(0)';
    }, 100 * index);
  });

  miniBars.forEach((bar, index) => {
    const targetWidth = bar.dataset.width || bar.style.width || '75%';
    bar.dataset.width = targetWidth;
    bar.style.width = '0';

    setTimeout(() => {
      bar.style.transition = 'width 0.8s ease';
      bar.style.width = targetWidth;
    }, 200 + index * 120);
  });

  const queueText = document.querySelector('.queue-header strong');
  const patientEl = document.querySelector('.book-info strong');

  if (queueText) {
    let current = 0;
    const values = ['18 mins', '12 mins', '8 mins', '5 mins'];
    setInterval(() => {
      current = (current + 1) % values.length;
      queueText.textContent = values[current];
    }, 2600);
  }

  async function refreshQueue() {
    try {
      const response = await fetch('/api/queue');
      const data = await response.json();

      if (!data || !data.patients || data.patients.length === 0) return;

      const firstPatient = data.patients[0];
      if (queueText) {
        queueText.textContent = `${Math.round(firstPatient.estimated_wait)} mins`;
      }

      if (patientEl) {
        patientEl.textContent = firstPatient.name;
      }

      const rows = document.querySelectorAll('.queue-row');
      data.patients.slice(0, rows.length).forEach((patient, index) => {
        const row = rows[index];
        if (!row) return;
        const name = row.querySelector('strong');
        const service = row.querySelector('small');
        const time = row.querySelector('span');

        if (name) name.textContent = patient.name;
        if (service) service.textContent = patient.service;
        if (time) time.textContent = `${Math.round(patient.estimated_wait)} mins`;
      });
    } catch (error) {
      console.error('Queue refresh failed:', error);
    }
  }

  refreshQueue();
  setInterval(refreshQueue, 5000);
});
